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
""" test_aclinit_config """
import json
import mindspore as ms
import os
from tests.device_utils import set_device
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_set_aclinit_config():
    """
    Feature: test aclinit_config
    Description: Test case for simplest aclinit_config setting
    Expectation: The results are as expected
    """
    set_device()
    ms.device_context.ascend.op_debug.aclinit_config({"max_opqueue_num": "20000", "err_msg_mode": "1"})
    try:
        with open("aclinit.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data["max_opqueue_num"] == "20000"
        assert data["err_msg_mode"] == "1"
    finally:
        if os.path.exists("aclinit.json"):
            os.remove("aclinit.json")
