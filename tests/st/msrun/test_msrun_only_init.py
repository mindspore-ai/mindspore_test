# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================

import time
import argparse

import mindspore.context as context
from mindspore.communication import init

parser = argparse.ArgumentParser(description='test_ps_lenet')
parser.add_argument("--device_target", type=str, default="GPU")
parser.add_argument("--dataset_path", type=str, default="/home/workspace/mindspore_dataset/mnist")
args, _ = parser.parse_known_args()
device_target = args.device_target
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)


if __name__ == "__main__":
    init()
    time.sleep(3)
    print("==== End initialize Communication group ====")
