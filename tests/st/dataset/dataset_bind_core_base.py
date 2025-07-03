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
dataset bind core
"""
import argparse
import numpy as np

import mindspore as ms
import mindspore.dataset as ds


class MyAccessible:
    def __init__(self):
        self._data = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def binding_function():
    """
    Verify Binding functions by NumpySlicesDataset and batch operation.
    """
    ms.runtime.set_cpu_affinity(True, [], {"minddata": [0, 1, 2, 3, 4, 5, 6, 7]})

    data = [1, 2, 3, 4, 5, 6]
    dataset = ds.NumpySlicesDataset(data=data, column_names=["data"])
    dataset = dataset.batch(2)
    for _ in dataset.create_tuple_iterator(num_epochs=1):
        continue


def numa_and_binding():
    """
    Verify that set_cpu_affinity and set_numa_enable are set at the same time.
    """
    ms.runtime.set_cpu_affinity(True, [], {"minddata": [0, 1, 2, 3, 4, 5, 6, 7]})
    ds.config.set_numa_enable(True)

    data = [1, 2, 3, 4, 5, 6]
    dataset = ds.NumpySlicesDataset(data=data, column_names=["data"])
    for _ in dataset.create_tuple_iterator(num_epochs=1):
        continue


def binding_python_process():
    """
    Verify Binding functions by GeneratorDataset and batch operation.
    """
    ms.runtime.set_cpu_affinity(True, [], {"minddata": [0, 1, 2, 3, 4, 5, 6, 7]})

    def batch_func(input_data, batch_info):
        return np.array(input_data)

    generator_dataset = ds.GeneratorDataset(source=MyAccessible(), column_names='col', python_multiprocessing=True,
                                            num_parallel_workers=2, shuffle=False)
    generator_dataset = generator_dataset.map(operations=[(lambda x: x - 1)], num_parallel_workers=2,
                                              python_multiprocessing=True)
    generator_dataset = generator_dataset.batch(batch_size=2, python_multiprocessing=True, num_parallel_workers=2,
                                                per_batch_map=batch_func)
    for _ in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        break


def binding_independent_dataset_process():
    """
    Verify the independent dataset process binding kernel functionality.
    """
    ms.runtime.set_cpu_affinity(True, [], {"minddata": [0, 1, 2, 3, 4, 5, 6, 7]})

    generator_dataset = ds.GeneratorDataset(source=MyAccessible(), column_names='col', python_multiprocessing=False,
                                            num_parallel_workers=2, shuffle=False)

    for _ in generator_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run specific functions.')
    parser.add_argument('function', type=str, help='Function to run (one or two)')

    args = parser.parse_args()

    if args.function == 'first_function':
        binding_function()
    elif args.function == 'second_function':
        numa_and_binding()
    elif args.function == 'third_function':
        binding_python_process()
    elif args.function == 'fourth_function':
        binding_independent_dataset_process()
