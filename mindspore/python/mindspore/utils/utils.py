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
"""mindspore utils."""
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore.common.api import jit_class


@jit_class
class ExitByRequest:
    """
    Gracefully exits the training process after get exit request.
    """

    def __init__(self):
        super(ExitByRequest, self).__init__()
        from mindspore.communication.management import get_group_size
        self.all_reduce = P.AllReduce()
        self.equal = P.Equal()
        self.assign = P.Assign()
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.is_distributed = _get_parallel_mode() != ParallelMode.STAND_ALONE
        if self.is_distributed:
            self.group_size = get_group_size()
            self.base = Tensor([self.group_size], dtype=mstype.int32)
        self.base1 = Tensor([1], mstype.int32)
        self.true = Tensor(True, mstype.bool_)

    def exit_by_request(self, grad, init_value, exit_value):
        """
        update GracefulExit flag by Assign op, the value is the output of AllReduce op
        :param grad: grad of net, or output of opt
        :param init_value: input value of AllReduce, a parameter
        :param exit_value: graceful exit value(out of AllReduce), update by Assign op
        :return: grad
        """
        if self.is_distributed:
            all_status = self.all_reduce(init_value)
            equal = self.equal(all_status, self.base)
            reduce_all = self.reduce_all(equal)
            grad = F.depend(grad, self.assign(exit_value, reduce_all))
        return grad
