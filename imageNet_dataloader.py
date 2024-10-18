import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import albumentations as A
import torch
import transformers
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import glob
import os
from tqdm import tqdm
import pandas as pd
import gc
import json
import numpy as np
import multiprocessing
import torchvision
from PIL import Image
import io



class Dataloader():

    def __init__(self, args, cuda_all_embeddings=None):
        """
        Args:
        args (dict): {
            files_path (str): path to files (e.g. "imagenet_1k_resized_256/data/train*.parquet")
            label_dict_path (str): path to label dict from https://github.com/anishathalye/imagenet-simple-labels
            image_processor_model_path (str): path to model to load the image-processor parameters from, for example resnet50
            store_images_compressed_on_cpu (bool): recommended for large dataset / train split. will decompress images on cpu during training. default True
            total_data_size (int): set to > 0 to truncate data samples to this amount, default -1
            cuda_loaded_max_size (int): number of samples at once on cuda, reducing this will reduce vram requirement. default 20_000
            region_count (int): number of blocks for updating images in cuda memory during training, should be >= 3. default 4
            max_process_workers (int): default 4
            verbose (bool): default True
        }
        """

        self.set_multiprocessing_start_method()

        if self._is_in_asyncio_loop():
            nest_asyncio.apply()

        self.executor = ThreadPoolExecutor(max_workers=4)
        
        for key, value in args.items():
            setattr(self, key, value)

        self.async_tensor_transfer_stream = torch.cuda.Stream()
        # self.region_count = 2 # now part of input args
        assert(self.cuda_loaded_max_size%self.region_count == 0)
        self.region_size = self.cuda_loaded_max_size // self.region_count
        assert(self.cuda_loaded_max_size % self.region_size == 0)
        assert(self.total_data_size % self.region_size == 0 or self.total_data_size == -1)
        
        self.region_lock_list = [False for _ in range(self.region_count)] # True if data transfer in progress

        self.image_processor_settings = transformers.AutoImageProcessor.from_pretrained(self.image_processor_model_path).__dict__
        self.text_vectorizer = get_clip_text_vectorizer()

        self.image_rescalefactor = torch.tensor(self.image_processor_settings['rescale_factor']).cuda()
        self.image_mean = torch.tensor(self.image_processor_settings['image_mean'])[None, :, None, None].cuda()
        self.image_std = torch.tensor(self.image_processor_settings['image_std'])[None, :, None, None].cuda() 

        self.cuda_all_embeddings = cuda_all_embeddings

        self.cpu_images, self.cpu_labels, self.cuda_all_embeddings = self.load_data()

        assert(len(self.cpu_images) == self.total_data_size)
        self.cuda_all_embeddings = self.cuda_all_embeddings.cuda()
        gc.collect()

        self.cuda_images = torch.zeros((self.cuda_loaded_max_size, 3, 224, 224), dtype=torch.uint8, device="cuda")
        self.cuda_labels = torch.zeros((self.cuda_loaded_max_size), dtype=torch.int32, device="cuda")

        self.total_size = len(self.cpu_images)
        self.current_step = 0 # which images have already been used this epoch
        self.current_region = 0
        self.last_loaded_index = 0
        self.shuffle_before_next_load = True
        self.epoch = 0
        self.wait_for_load_notification = True

        self.was_used = False

        def start_loop(loop):
            # Set the event loop for the current thread
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.process_pool, self.loop, self.t = self._start_parallel_stuff()

        # initial load
        for i in range(self.region_count):
            self.load_next_region()

        self.verbose=False
        self.wait_for_load_notification=True

    

    def _is_in_asyncio_loop(self):
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def set_multiprocessing_start_method(self):
        try:
            multiprocessing.set_start_method('spawn', force=False)
        except RuntimeError as e:
            if 'context has already been set' in str(e):
                # Context has already been set, we can safely pass or log this event
                pass
            else:
                # Re-raise any unexpected RuntimeError
                raise

    def _start_parallel_stuff(self):
        
        def start_loop(loop):
            # Set the event loop for the current thread
            asyncio.set_event_loop(loop)
            loop.run_forever()

        if self.store_images_compressed_on_cpu:
            process_pool = ProcessPoolExecutor(max_workers=self.max_process_workers)
        else:
            process_pool = None
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=start_loop, args=(loop,))
        t.start()

        return process_pool, loop, t

    def reset(self):
        self._stop_parallel_stuff()
        
        self.total_size = len(self.cpu_images)
        self.current_step = 0
        self.current_region = 0
        self.last_loaded_index = 0
        self.epoch = 0
        self.shuffle_before_next_load = True

        self.was_used = False

        self.region_lock_list = [False for _ in range(self.region_count)]

        self.process_pool, self.loop, self.t = self._start_parallel_stuff()
        
        for i in range(self.region_count):
            self.load_next_region()

    async def _handle_future(self, future):
        try:
            result = await future
        except Exception as e:
            print(f"Exception in background task: {e}")

    def _run_in_background(self, func, *args):
        coroutine = func(*args)
        self.loop.create_task(coroutine)
    
    def run_in_background(self, func, *args):
        self.loop.call_soon_threadsafe(asyncio.create_task, func(*args))
    
    def load_next_region(self):
        
        self.run_in_background(self.async_load_to_cuda, self.current_region, self.last_loaded_index, self.shuffle_before_next_load)
        self.shuffle_before_next_load = False
        
        self.last_loaded_index += self.region_size
        self.current_region += 1
        
        if self.current_region >= self.region_count:
            self.current_region = self.current_region % self.region_count
            
        if self.last_loaded_index >= self.total_data_size:
            self.last_loaded_index = self.last_loaded_index % self.total_data_size # new epoch
            self.shuffle_before_next_load = True
    
    def get_batch(self, batch_size, num_false_embeddings=3, augmentations=None):
        """
        returns:
        image input: images_batch
        embedding input: embeddings_batch
        loss true: true_batch
        """

        assert(batch_size*3 < self.region_size)
        self.was_used = True
        
        batch_regions = set() # which tensor regions the data is retrieved from
        batch_regions.add((self.current_step//self.region_size)%self.region_count)
        batch_regions.add(((self.current_step+batch_size)//self.region_size)%self.region_count)
        batch_regions = list(batch_regions)

        # wait if trying to load from a region that is being updated
        if any(self.region_lock_list[i] for i in batch_regions):
            if self.wait_for_load_notification: print(f"waiting for regions {batch_regions} to all be unlocked. current region_lock_list: {self.region_lock_list}")
            while any(self.region_lock_list[i] for i in batch_regions):
                time.sleep(0.001)
            if self.wait_for_load_notification: print(f"relevant regions are unlocked, continuing", flush=True)
            

        if len(batch_regions) == 1 and batch_regions[0] != self.current_region:
            self.load_next_region()
        
        images_batch = self.load_tensor(self.cuda_images, self.current_step, self.current_step+batch_size) # images[self.current_step:self.current_step+batch_size].detach()
        labels_batch = self.load_tensor(self.cuda_labels, self.current_step, self.current_step+batch_size) # labels[self.current_step:self.current_step+batch_size].detach()
        self.current_step += batch_size
        if self.current_step >= self.total_data_size:
            self.current_step = self.current_step % self.total_data_size
            self.epoch += 1

        images_batch = images_batch.to(torch.float32) / 255
        if augmentations != None:
            images_batch = augmentations(images_batch)
            images_batch = torch.clamp(images_batch, 0.0, 1.0)

        images_batch = images_batch * 255
        images_batch = self.final_image_processing(images_batch)
        embeddings_batch = self.cuda_all_embeddings[labels_batch].detach().unsqueeze(1) # [batch, 1, embedding_dim]
        embeddings_batch = self.embeddings_add_false(embeddings_batch, labels_batch, num_false_embeddings)
        true_batch = torch.zeros((labels_batch.shape[0], 1 + num_false_embeddings), device="cuda").detach()
        true_batch[:, 0] = 1 # the first embedding is always the true one, all others are the random false ones

        return images_batch, embeddings_batch, true_batch

    def load_tensor(self, tensor, start, end):
        start = start % self.cuda_loaded_max_size
        end = end % self.cuda_loaded_max_size

        if start < end:
            return tensor[start:end].detach()
        else:
            return torch.concatenate([tensor[start:].detach(), tensor[:end].detach()], dim=0)

    async def async_load_to_cuda(self, cuda_region, cpu_start, shuffle_before_load=False):
        try:    
                
            if shuffle_before_load:
                self.shuffle()
            
            assert(self.region_lock_list[cuda_region] == False)
            self.region_lock_list[cuda_region] = True
            
            cuda_start = self.region_size * cuda_region
            cuda_end = self.region_size * (cuda_region+1)
    
            cpu_end = cpu_start + self.region_size
    
            if self.verbose: print(f"loading cuda region {cuda_region} ({cuda_start}-{cuda_end}) from cpu ({cpu_start}-{cpu_end})", flush=True)

            if not self.store_images_compressed_on_cpu:
                if cpu_start < cpu_end:
                    cpu_images = self.cpu_images[cpu_start:cpu_end]
                    cpu_labels = self.cpu_labels[cpu_start:cpu_end]
                else:
                    cpu_images = torch.concatenate([self.cpu_images[cpu_start:], self.cpu_images[:cpu_end]], dim=0)
                    cpu_labels = torch.concatenate([self.cpu_labels[cpu_start:], self.cpu_labels[:cpu_end]], dim=0)
                    
            else:
                if cpu_start < cpu_end:
                    try:
                        cpu_images = await self.loop.run_in_executor(self.process_pool, pickleable_images_bytes_to_preprocessed_tensor, self.cpu_images[cpu_start:cpu_end], self.image_processor_settings)
                    except Exception as e:
                        print(f"ERROR: {e}")
                    cpu_labels = self.cpu_labels[cpu_start:cpu_end]
                else:
                    processed_images = await asyncio.gather(
                        self.loop.run_in_executor(self.process_pool, pickleable_images_bytes_to_preprocessed_tensor, self.cpu_images[cpu_start:], self.image_processor_settings),
                        self.loop.run_in_executor(self.process_pool, pickleable_images_bytes_to_preprocessed_tensor, self.cpu_images[:cpu_end], self.image_processor_settings)
                    )
                    cpu_images = torch.cat(processed_images, dim=0)
                    cpu_labels = torch.concatenate([self.cpu_labels[cpu_start:], self.cpu_labels[:cpu_end]], dim=0)


            with torch.cuda.stream(self.async_tensor_transfer_stream):
                self.cuda_images[cuda_start:cuda_end] = cpu_images.cuda(non_blocking=True)
                self.cuda_labels[cuda_start:cuda_end] = cpu_labels.cuda(non_blocking=True)
        
            self.async_tensor_transfer_stream.synchronize()
            
            assert(self.region_lock_list[cuda_region] == True)
            self.region_lock_list[cuda_region] = False
            
            if self.verbose: print(f"done loading cuda region {cuda_region}.", flush=True)
        except Exception as e:
            print(e, flush=True)
            raise(e)

    def shuffle(self):
        if self.verbose: print("new epoch (starting from beginning of cpu data), shuffling.", flush=True)
        shuffle_indices = torch.randperm(self.total_data_size)
        try:
            if torch.is_tensor(self.cpu_images):
                self.cpu_images = self.cpu_images[shuffle_indices]
            else:
                self.cpu_images = np.array(self.cpu_images, dtype=object)
                self.cpu_images = self.cpu_images[shuffle_indices.numpy()]
                self.cpu_images = self.cpu_images.tolist()
        except Exception as e:
            print(e, flush=True)
            raise(e)
        self.cpu_labels = self.cpu_labels[shuffle_indices]

    def final_image_processing(self, image_tensor):
        image_tensor = ((image_tensor * self.image_rescalefactor) - self.image_mean) / self.image_std
        
        return image_tensor

    def embeddings_add_false(self, embeddings_batch, labels_batch, num_false_embeddings):
        random_indexes = torch.randint(low=0, high=999, size=(labels_batch.shape[0], num_false_embeddings), device="cuda")
        random_indexes += (random_indexes >= labels_batch.unsqueeze(1)) # shift all values that are >= class_label by +1 to avoid it in the random generation
        random_false_embeddings = self.cuda_all_embeddings[random_indexes].detach()
        embeddings_batch = torch.concatenate([embeddings_batch, random_false_embeddings], dim=1)
    
        return embeddings_batch

    def images_bytes_to_preprocessed_tensor(self, image_list):
        pil_to_tensor = torchvision.transforms.PILToTensor()
        resize_shortest_edge = int(self.image_processor_settings['size']['shortest_edge'] / self.image_processor_settings['crop_pct'])
        resize = torchvision.transforms.Resize(resize_shortest_edge, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        center_crop = torchvision.transforms.CenterCrop(self.image_processor_settings['size']['shortest_edge'])

        for i in range(len(image_list)):
            x = image_list[i]
            x = Image.open(io.BytesIO(x['bytes']))
            x = pil_to_tensor(x)
            x = resize(x)
            x = center_crop(x)
            image_list[i] = x

        return image_list

    async def async_images_bytes_to_preprocessed_tensor(self, image_list):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, self.images_bytes_to_preprocessed_tensor, image_list)
            result = torch.stack(result)
        except Exception as e:
            print(e, flush=True)
            raise(e)
        return result

    def get_all_label_embeddings_tensor(self):
        with open(self.label_dict_path, "r") as file:
            labels_to_text_dict = json.loads(file.read())
            
        all_label_embeddings = self.text_vectorizer(labels_to_text_dict) # torch.Size([1000, 1, 512])
        all_label_embeddings = all_label_embeddings.squeeze(1)

        return all_label_embeddings

    def load_data(self):
        """
        loads data from self.files_path

        returns:
        merged tensor images
        merged tensor classes
        merged tensor class -> text embedding
        """
        if isinstance(self.files_path, str):
            paths = glob.glob(self.files_path)
            paths = [path for path in paths if os.path.getsize(path) > 1000] # filter out pointer files that are not downloaded
    
            if self.verbose: print("loading files", flush=True)
                
            def read_parquet_file(path):
                return pd.read_parquet(path, engine='pyarrow')

            n_threads = 4

            # Reading files in parallel
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                dfs = list(tqdm(executor.map(read_parquet_file, paths), total=len(paths)))
                
            df = pd.concat(dfs)
    
            if self.verbose: print("shuffling and reducing to target size", flush=True)
            df = df.sample(frac=1)
            if self.total_data_size == -1:
                self.total_data_size = len(df) - (len(df) % self.cuda_loaded_max_size)
                print(f"total_data_size == -1, set to max divisible by cuda_loaded_max_size without rest: {self.total_data_size}", flush=True)
            df = df.iloc[:self.total_data_size]
            gc.collect()

        else:
            # assume provided object is df
            df = self.files_path

        if self.verbose: print("extracting images and labels from df and deleting df", flush=True)
        col_images = df['image'].tolist() # currently in bytes format (str)
        col_labels = df['label'].tolist() # currently in int format
        del df
        gc.collect()

        if not self.store_images_compressed_on_cpu:
            if self.verbose: print("converting images to preprocessed tensors", flush=True)
            col_images = torch.stack(self.images_bytes_to_preprocessed_tensor(col_images))

        if self.verbose: print("converting labels to tensors", flush=True)
        col_labels = torch.stack([torch.tensor(entry) for entry in tqdm(col_labels)])
        col_images = col_images.contiguous() if not self.store_images_compressed_on_cpu else col_images
        col_labels = col_labels.contiguous()

        if self.cuda_all_embeddings == None:
            if self.verbose: print("creating class -> text embedding tensor", flush=True)
            all_labels = self.get_all_label_embeddings_tensor()
            all_labels = all_labels.contiguous()
        else:
            assert torch.is_tensor(self.cuda_all_embeddings)
            if self.verbose: print("class -> text embedding tensor was provided, not creating new", flush=True)
            all_labels = self.cuda_all_embeddings

        return col_images, col_labels, all_labels

    def __len__(self):
        return len(self.cpu_images)

    def _stop_parallel_stuff(self):
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except:
            pass
        try:
            self.t.join()
        except:
            pass
        try:
            self.process_pool.shutdown(wait=True)
        except:
            pass

    def __del__(self):
        self._stop_parallel_stuff()

    def undo_final_processing(self, image_tensor):
        image_tensor = ((image_tensor * self.image_std) + self.image_mean) / self.image_rescalefactor
        return image_tensor
        

def pickleable_images_bytes_to_preprocessed_tensor(image_list, image_processor_settings):
    pil_to_tensor = torchvision.transforms.PILToTensor()
    resize_shortest_edge = int(image_processor_settings['size']['shortest_edge'] / image_processor_settings['crop_pct'])
    resize = torchvision.transforms.Resize(resize_shortest_edge, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    center_crop = torchvision.transforms.CenterCrop(image_processor_settings['size']['shortest_edge'])

    for i in range(len(image_list)):
        x = image_list[i]
        x = Image.open(io.BytesIO(x['bytes']))
        x = pil_to_tensor(x)
        x = resize(x)
        x = center_crop(x)
        image_list[i] = x

    images_tensor = torch.stack(image_list)
    
    return images_tensor



class TextVectorizer(nn.Module):
    def __init__(self, tokenizer, embedder_text, projector_text):
        super(TextVectorizer, self).__init__()
        self.tokenizer = tokenizer
        self.embedder_text = embedder_text.cuda()
        self.projector_text = projector_text.cuda()

    def forward(self, x):
        # print("text vectorizer input:", x)
        tokens = self.tokenizer(x).input_ids

        merker = []
        for token_seq in tokens:
            token_seq = torch.tensor(token_seq, device="cuda").unsqueeze(0)
            embedding = self.embedder_text(token_seq).pooler_output
            merker.append(embedding)

        x = torch.stack(merker)
            
        x = self.projector_text(x)
        return x

def get_clip_text_vectorizer():
    clip_model = CLIPModel.from_pretrained("largefiles/clip-vit-base-patch32/")
    clip_processor = CLIPProcessor.from_pretrained("largefiles/clip-vit-base-patch32", clean_up_tokenization_spaces=True, do_rescale=True)
    text_vectorizer = TextVectorizer(clip_processor.tokenizer, clip_model.text_model, clip_model.text_projection).eval()
    
    return text_vectorizer