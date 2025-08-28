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
"""HSDP communication interface"""
from mindspore.communication import create_group
from mindspore import ops
HSDP_GROUP_CACHE = set()


class HSDPComm:
    """
    HSDP communication interface.
    """
    def create_group(self, group_name, rank_list):
        """create communication group with group name and rank list."""
        if group_name in HSDP_GROUP_CACHE:
            return
        create_group(group_name, rank_list)
        HSDP_GROUP_CACHE.add(group_name)

    def all_gather(self, group_name, data):
        """all gather data with group group_name."""
        return ops.AllGather(group=group_name)(data)

    def reduce_scatter(self, group_name, data):
        """reduce scatter data with group group_name."""
        return ops.ReduceScatter(group=group_name)(data)

    def all_reduce(self, group_name, data):
        """all reduce data with group group_name."""
        return ops.AllReduce(group=group_name)(data)
