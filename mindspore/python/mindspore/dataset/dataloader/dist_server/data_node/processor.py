# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Data preprocessor.
"""


import mindspore
import cv2
import numpy as np
import albumentations as A
from transformers import AutoTokenizer
import os
import sys

class DataProcessor:
    def __init__(self, tokenizer_path, df_ref):
        self.df = df_ref
        self.image_transform = A.Compose([
            A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
            A.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        ])
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.text_max_length = self.tokenizer.model_max_length
        print(f"[Processor] Initialized with {len(self.df)} samples.")

    def get_item(self, index):
        try:
            row = self.df.iloc[index]
            image_path = row['image_path']
            caption = row['caption']
            
            img_bytes = open(image_path, 'rb').read()
            img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  
            transformed = self.image_transform(image=img_rgb)
            image_np = transformed['image']         
            image_tensor = mindspore.Tensor(image_np.transpose(2, 0, 1), dtype=mindspore.float32)
            
            tokenized_output = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt"
            )            
            input_ids = tokenized_output['input_ids'].squeeze(0)
            attention_mask = tokenized_output['attention_mask'].squeeze(0)
            return image_tensor, input_ids, attention_mask   
             
        except Exception as e:
            worker_pid = os.getpid()
            print(f"--- [PID {worker_pid}] !!! ERROR IN GET_ITEM !!!", file=sys.stderr, flush=True)
            print(f"    Index: {index} | Path: {image_path}", file=sys.stderr, flush=True)
            print(f"    Error: {e}", file=sys.stderr, flush=True)
            return None, None, None


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

        if not batch_images:
            return None

        return {
            "images": mindspore.ops.stack(batch_images),
            "input_ids": mindspore.ops.tack(batch_input_ids),
            "masks": mindspore.ops.stack(batch_masks)
        }