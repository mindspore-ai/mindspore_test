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
"""dryrun."""
import types
import traceback
import os
from mindspore._c_expression import Tensor as Tensor_
from mindspore.common import Tensor
from mindspore import log as logger
from mindspore.common._stub_tensor import StubTensor
from mindspore.common import dtype as mstype
from mindspore._checkparam import is_stub_tensor

class TraceBack():
    """
    traceback warning logs in dryrun mode
    """
    def __init__(self):
        self.stack_str_set = set()
    def inject(self, method):
        """
        inject warning logs in dryrun mode
        """
        def new_method(*args, **kwargs):
            stack_list = traceback.format_list(traceback.extract_stack())
            stack_str = "".join(stack_list)
            if "Parameter" not in stack_str and stack_str not in self.stack_str_set:
                self.stack_str_set.add(stack_str)
                logger.warning("In dryrun mode, you cannot obtain real tensor value, and the traceback is {%s}",
                               stack_list)
            return method(*args, **kwargs)
        return new_method

def no_inject_traceback_for_print(self):
    if is_stub_tensor(self):
        self = self.stub_sync()
    if self.dtype == mstype.type_none:
        return "Unknown Tensor type!"
    if self.has_init:
        self.init_data()
    return str(Tensor_.asnumpy(self))


def set_simulation():
    """
    This interface is used to enable the dryrun function. The dryrun function is mainly used to simulate the actual
    operation of the large model. After it is enabled, the memory usage, compilation information, etc. can be simulated
    without occupying device card. In the pynative mode, once it is enabled, if values are fetched from the device to
    the host, the Python call stack log will be printed to inform users that these values are inaccurate.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    obj = TraceBack()
    Tensor.asnumpy = obj.inject(Tensor.asnumpy)
    Tensor.is_contiguous = obj.inject(Tensor.is_contiguous)
    Tensor.flush_from_cache = obj.inject(Tensor.flush_from_cache)
    StubTensor.asnumpy = obj.inject(StubTensor.asnumpy)
    StubTensor.is_contiguous = obj.inject(StubTensor.is_contiguous)
    StubTensor.flush_from_cache = obj.inject(StubTensor.flush_from_cache)
    Tensor.__str__ = no_inject_traceback_for_print
    StubTensor.__str__ = no_inject_traceback_for_print


def mock(mock_val, *args):
    """
    If `if xxx: ` in the network need to use the actual execution values which cannot be obtained through dryrun mode,
    this interface can be used to return static simulated values.

    Inputs:
        - **mock_val** (Union[value, Tensor]): The value you want to return.
        - **args**:
    Outputs:
        If set_simulation, the mock_val will be returned; otherwise, the actual execution value
        of args will be returned.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> dryrun.set_simulation()
        >>> a = ms.Tensor(np.random.rand(3, 3).astype(np.float32))
        >>> if dryrun.mock(True, a[0, 0] > 0.5):
        ...     print("return mock_val: True.")
        return mock_val: True

        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, a[0, 0] > 0.5):
        ...     print("return real execution: True.")
        return real execution: True.

        >>> import mindspore as ms
        >>> from mindspore.utils import dryrun
        >>> import numpy as np
        >>> a = ms.Tensor(np.ones((3, 3)).astype(np.float32))
        >>> if dryrun.mock(False, (a > 0.5).any):
        ...     print("return real execution: True.")
        return real execution: True.
    """
    if os.environ.get('MS_SIMULATION_LEVEL'):
        return mock_val
    if len(args) == 1:
        if isinstance(args[0], types.MethodType):
            return args[0]()
        return args[0]
    return args[0](*args[1:])
