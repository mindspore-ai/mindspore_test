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

import time

import numpy as np

from mindspore.dataset.dataloader import DataLoader, Dataset


class MockDataset(Dataset):

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        time.sleep(5)
        return np.array(self.data[index], dtype=np.uint8)

    def __len__(self):
        return self.num_samples


def run_data_loader():
    data_loader = DataLoader(MockDataset(100), num_workers=8)
    for _ in data_loader:
        pass


if __name__ == "__main__":
    run_data_loader()
