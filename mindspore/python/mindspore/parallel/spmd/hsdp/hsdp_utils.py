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
"""HSDP optimizer shared level"""
from enum import auto, Enum


class OptimizerLevel(Enum):
    """
        Optimizer level:
                - SHARD_OPT:
                  Splitting is performed on optimizer state.
                - SHARD_OPT_GRAD:
                  Splitting is performed on optimizer state, and gradients.
                - SHARD_OPT_GRAD_PARAM:
                  Splitting is performed on optimizer state, gradients and weights.
    """
    SHARD_OPT = auto()
    SHARD_OPT_GRAD = auto()
    SHARD_OPT_GRAD_PARAM = auto()


class HSDPConfig:
    """HSDP config"""

    def __init__(self, shard_size, threshold, requires_acc_grad, shard_level, use_cell_hook, reduce_dtype):
        """
            HSDP config init method
            Args:
                shard_size: optimizer weight sharded size.
                threshold: minimum weight size to shard.
                requires_acc_grad: requires gradient accumulation.
                shard_level: optimizer shard level.
                use_cell_hook: use cell hook or param hook for hsdp implement.
                reduce_dtype: set gradient reduce dtype.
        """
        self.shard_size = shard_size
        self.threshold = threshold
        self.requires_acc_grad = requires_acc_grad
        self.shard_level = shard_level
        self.use_cell_hook = use_cell_hook
        self.reduce_dtype = reduce_dtype
