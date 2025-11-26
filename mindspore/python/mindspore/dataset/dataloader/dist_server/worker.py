import ray
import pandas as pd
import torch
import torch_npu
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
import os
import sys



@ray.remote(num_cpus=1)
class DataWorkerActor:
    
    def __init__(self, dataset_params, df_ref):
        worker_pid = os.getpid()
        try:
            self.df = df_ref 
            self.image_transform = A.Compose([
                A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], 
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
                ToTensorV2() 
            ])

            local_tokenizer_path = dataset_params.get("tokenizer_path")
            if not local_tokenizer_path:
                print(f"--- [PID {worker_pid}] CRITICAL ERROR: 'tokenizer_path' not found!", flush=True)
                raise ValueError("Worker: 'tokenizer_path' not found in dataset_params.")

            self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path) 
            self.text_max_length = self.tokenizer.model_max_length
            print(f"--- [PID {worker_pid}] INIT: SUCCESS! Worker is ready. ---", flush=True)

        except Exception as e:
            print(f"--- [PID {worker_pid}] !!! PYTHON EXCEPTION IN INIT !!!", file=sys.stderr, flush=True)
            print(f"    Error: {e}", file=sys.stderr, flush=True)
            raise e
            
            
            
    def get_item(self, index):
        try:
            row = self.df.iloc[index]
            image_path = row['image_path']
            caption = row['caption']
            
            img_bytes = open(image_path, 'rb').read()
            img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)           
            image_tensor = self.image_transform(image=img_rgb)['image']
            
            tokenized_output = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized_output['input_ids'].squeeze(0)
            attention_mask = tokenized_output['attention_mask'].squeeze(0)

            return (image_tensor, input_ids, attention_mask)

        
        except Exception as e:
            worker_pid = os.getpid()
            print(f"--- [PID {worker_pid}] !!! ERROR IN GET_ITEM !!!", file=sys.stderr, flush=True)
            print(f"    Index: {index} | Path: {image_path}", file=sys.stderr, flush=True)
            print(f"    Error: {e}", file=sys.stderr, flush=True)
            return (None, None, None)



    def get_batch(self, indices):
        batch_images = []
        batch_input_ids = []
        batch_masks = []
        
        for idx in indices:
            images,input_ids,masks = self.get_item(index=idx)
            if images is not None:
                batch_images.append(images)
                batch_input_ids.append(input_ids)
                batch_masks.append(masks)         

        if len(batch_images) == 0:
            return None, None, None
        final_images = torch.stack(batch_images)       # (B, C, H, W)
        final_input_ids = torch.stack(batch_input_ids) # (B, L)
        final_masks = torch.stack(batch_masks)         # (B, L)
        
        return final_images, final_input_ids, final_masks       
        
    