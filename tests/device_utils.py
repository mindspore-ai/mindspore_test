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
# ============================================================================
"""
All the following Python APIs are only designed for access ci testing
"""
import os
import mindspore as ms

def set_device():
    devcie_target = os.getenv("DEVICE_TARGET")
    device_id = os.getenv("DEVICE_ID")
    if device_id is None:
        ms.set_device(devcie_target)
    else:
        ms.set_device(devcie_target, int(device_id))


def get_device():
    devcie_target = os.getenv("DEVICE_TARGET")
    return devcie_target


def get_device_id():
    return os.getenv("DEVICE_ID", "0")
