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
"""
Collective communication interface.

Note that the APIs in the following list need to preset communication environment variables.

For Ascend devices, it is recommended to use the msrun startup method
without any third-party or configuration file dependencies.
Please see the `msrun start up
<https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_
for more details.
"""
from mindspore.mint.distributed.distributed import init_process_group, destroy_process_group, get_rank, get_world_size


__all__ = [
    "init_process_group", "destroy_process_group", "get_rank", "get_world_size"
]
