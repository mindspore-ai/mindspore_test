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

import pytest

import mindspore.dataset as ds


def test_dataloader():
    dataset = ds.Dataset()
    dataloader = ds.DataLoader(dataset)


class MySampler:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            data = self.index
            self.index += 1
            return data
        else:
            raise StopIteration


class MyDataset(ds.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def test_dataloader_single_process_iteration():
    dataset = MyDataset(10)
    sampler = MySampler(5)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    for data in dataloader:
        print(data)


# def test_dataloader_multiprocess_num_worker_warning():
#     dataset = ds.Dataset()
#     dataloader = ds.DataLoader(dataset, num_workers=64)
#     for data in dataloader:
#         break
