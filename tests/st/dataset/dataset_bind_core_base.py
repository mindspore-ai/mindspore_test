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

import mindspore as ms
import mindspore.dataset as ds


def binding_function():
    """
    Verify Binding functions by NumpySlicesDataset and batch operation.
    """
    ms.runtime.set_cpu_affinity(True)

    data = [1, 2, 3, 4, 5, 6]
    dataset = ds.NumpySlicesDataset(data=data, column_names=["data"])
    dataset = dataset.batch(2)
    for _ in dataset.create_tuple_iterator(num_epochs=1):
        continue


def numa_and_binding():
    """
    Verify that set_cpu_affinity and set_numa_enable are set at the same time.
    """
    ms.runtime.set_cpu_affinity(True)
    ds.config.set_numa_enable(True)

    data = [1, 2, 3, 4, 5, 6]
    dataset = ds.NumpySlicesDataset(data=data, column_names=["data"])
    for _ in dataset.create_tuple_iterator(num_epochs=1):
        continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run specific functions.')
    parser.add_argument('function', type=str, help='Function to run (one or two)')

    args = parser.parse_args()

    if args.function == 'first_function':
        binding_function()
    elif args.function == 'second_function':
        numa_and_binding()
