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
Simple dataset pipeline
"""

import numpy as np
import sys

import mindspore.dataset as ds

def dataset_pipeline(python_multiprocessing):
    """dataset pipeline"""
    class GeneratorData:
        def __init__(self):
            self._data = np.random.uniform(low=600, high=800, size=(1000, 2))
            self._label = np.random.uniform(low=600, high=800, size=(1000, 1))

        def __getitem__(self, index):
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    dataset = ds.GeneratorDataset(GeneratorData(), column_names=["data", "label"], num_parallel_workers=2,
                                  python_multiprocessing=python_multiprocessing)
    dataset = dataset.filter(predicate=lambda label: label > 700, input_columns=["label"], num_parallel_workers=2)
    def map_func(data):
        return (data,)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=python_multiprocessing)
    def batch_func(data, _):
        return (data,)
    dataset = dataset.batch(2, num_parallel_workers=4, per_batch_map=batch_func, input_columns=["label"],
                            python_multiprocessing=python_multiprocessing)
    dataset = dataset.repeat()
    for _ in dataset.create_dict_iterator(output_numpy=True):
        pass

if __name__ == "__main__":
    if sys.argv[1] == "True":
        dataset_pipeline(True)
    elif sys.argv[1] == "False":
        dataset_pipeline(False)
    else:
        raise RuntimeError("Invalid parameter.")
