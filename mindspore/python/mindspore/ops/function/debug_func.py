# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Operators for debug function."""

from mindspore.ops.operations.debug_ops import Print
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .._primitive_cache import _get_cache_prim


def print_(*input_x):
    """
    Outputs the inputs to stdout. The outputs are printed to screen by default.
    It can also be saved in a file by setting the parameter  `print_file_path` in `context`.
    Once set, the output will be saved in the file specified by print_file_path.
    :func:`mindspore.parse_print` can be employed to reload the data.
    For more information, please refer to :func:`mindspore.set_context` and :func:`mindspore.parse_print`.
    In Ascend platform with graph mode, can set environment variables `MS_DUMP_SLICE_SIZE` and `MS_DUMP_WAIT_TIME`
    to solve operator execution failure when outputting big tensor or outputting tensor intensively.

    Note:
        In pynative mode, please use python print function.
        In Ascend platform with graph mode, the bool, int and float would be converted into Tensor to print, and
        str remains unchanged.
        This function is used for debugging.

    Args:
        input_x (Union[Tensor, bool, int, float, str, tuple, list]): The inputs of print_.
            Supports multiple inputs which are separated by ','.

    Returns:
        Invalid value, should be ignored.

    Raises:
        TypeError: If `input_x` is not one of the following: Tensor, bool, int, float, str, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.ones([2, 1]).astype(np.int32))
        >>> y = Tensor(np.ones([2, 2]).astype(np.int32))
        >>> result = ops.print_('Print Tensor x and Tensor y:', x, y)
        Print Tensor x and Tensor y:
        Tensor(shape=[2, 1], dtype=Int32, value=
        [[1],
         [1]])
        Tensor(shape=[2, 2], dtype=Int32, value=
        [[1, 1],
         [1, 1]])
    """
    print_op = _get_cache_prim(Print)()
    return print_op(*input_x)



def tensordump(file_name, tensor, mode='out'):
    """
    Save Tensor in numpy's npy format.

    In Parallel situation, tensordump will dump slice of data at each rank.

    In Ascend platform with graph mode,
    Your code OpA --> OpB may compiled as OpA --> RedistributionOps --> OpB.

    Note: The redistribution operator is introduced,
    Due to inter-device communication and shard strategies in the static graph parallel scenario.

    In case of OpA --> OpB, the dump data of OpA's output is equal to OpB's input.

    But in case of OpA --> RedistributionOps --> OpB,
    The dump data of OpA's output is not equal to OpB's input (Due to the redistribution operators).
    So the parameter mode is to handle this situation.

    Assuming OpA's output is used as both tensordump's input parameter and OpB's input parameter.
    Different requirements of saving dump data can be achieved by configuring parameter mode:

    - If the mode is 'out', the dump data contains only OpA's output slice.
    - If the mode is 'all', the dump data contains both OpA's output slice and OpB's input slice.
    - If the mode is 'in', the dump data contains only OpB's input slice.

    For mode 'all' or 'in', the input slice npy file format is: id_fileName_cNodeID_dumpMode_rankID.npy.

    For mode 'out' or 'all' the output slice npy file format is: id_filename.npy.

    - id: An auto increment ID.
    - fileName: Value of the parameter file_name
      (if parameter file_name is a user-specified path, the value of fileName is the last level of the path).
    - cNodeID: The cnode ID in ir graph of step_parallel_end.ir.
    - dumpMode: Value of the parameter mode.
    - rankID: Logical device id.

    Note:
        - The operator of tensordump doesn't support in control flow.
        - If current parallel mode is STAND_ALONE, mode should only be 'out'.
        - Parameter mode will be set to 'out' if user doesn't configure it.
        - This function is used for debugging.

    Args:
        file_name (str): The path of the npy file saves.
        tensor (Tensor): The tensor that user want to dump.
        mode (str, optional): Used to control tensordump behavior, available value is one of ['in', 'out', 'all'].
            Default value is ``out``.

    Raises:
        TypeError: If `file_name` is not str.
        TypeError: If `tensor` is not Tensor.
        TypeError: If `mode` is not str.
        ValueError: If `mode` is not in one of ['in', 'out', 'all'].

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Using msrun command to run below example: msrun --worker_num=2 --local_worker_num=2 --master_port=11450
            --log_dir=msrun_log --join=True --cluster_time_out=300 tensordump_example.py

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor, ops, context
        >>> from mindspore.ops import operations as P
        >>> from mindspore.communication import init, get_rank
        >>> init()
        >>> rank_id = get_rank()
        >>> dump_path = f'dumps/rank_{rank_id}/mul1_mul2.npy'
        >>> class Net(nn.Cell):
        ...     def __init__(self, strategy1, strategy2):
        ...         super(Net, self).__init__()
        ...         self.matmul1 = P.MatMul().shard(strategy1)
        ...         self.matmul2 = P.MatMul().shard(strategy2)
        ...
        ...     def construct(self, x, y, b):
        ...         out1 = self.matmul1(x, y)
        ...         ops.tensordump(dump_path, out1, 'all')
        ...         out2 = self.matmul2(out1, b)
        ...         return out2
        ...
        >>> ms.set_context(mode=ms.GRAPH_MODE, save_graphs=2)
        >>> context.set_auto_parallel_context(parallel_mode='semi_auto_parallel', full_batch=True)
        >>> strategy1 = ((1, 2), (2, 1))
        >>> strategy2 = ((1, 2), (2, 1))
        >>> net = Net(strategy1, strategy2)
        >>> x = Tensor(0.1 * np.random.randn(64, 64).astype(np.float32))
        >>> y = Tensor(0.1 * np.random.randn(64, 64).astype(np.float32))
        >>> b = Tensor(0.1 * np.random.randn(64, 64).astype(np.float32))
        >>> out = net(x, y, b)
        >>> print(f"out shape is: {out.shape}")
        >>> matmul1_output_slice = np.load('0_mul1_mul2.npy')                       # load matmul1's output slice
        >>> matmul2_input_slice = np.load('1_mul1_mul2_CNode_64_all_rank_0.npy')    # load matmul2's input slice
    """
    if not isinstance(file_name, str):
        raise TypeError(f"Parameter file_name should only be build_in str type but got: {type(file_name)}")
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Parameter tensor should only be Tensor type, but got: {type(tensor)}")
    if not isinstance(mode, str):
        raise TypeError(f"Parameter mode should only be build_in str type, but got: {type(mode)}")
    mode_list = ['out', 'in', 'all']
    if mode not in mode_list:
        raise ValueError(f"Parameter mode should in {mode_list}, but got {mode}")
    _tensordump = _get_cache_prim(P.TensorDump)(input_output=mode)
    return _tensordump(file_name, tensor)

__all__ = ['print_', 'tensordump']

__all__.sort()
