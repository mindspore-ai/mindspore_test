# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Defines parameter operators with functional form."""
import mindspore as ms
from mindspore import context
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.common.tensor import Tensor
from mindspore.communication.management import get_group_size, get_rank
from mindspore.parallel.shard import Layout, _DistributedTensorInfo
from mindspore.parallel._auto_parallel_context import _get_all_auto_parallel_context, _recover_auto_parallel_context

REDIST_CELL_CACHE = {}

# pylint: disable=W0212
def reshard(tensor, layout):
    r"""
    Specify the tensor by the given layout. The given layout must be type mindspore.Layout,
    can check :class:`mindspore.Layout` for reference.

    - In the Graph mode, this function can set the sharding propagation strategy of a tensor.
      For those tensor do not manually be set, their strategies are decided by the sharding
      strategy propagation algorithm automatically.
    - In the PyNative mode, this function can set a tensor sharding strategy in a Cell that
      runs in the Graph mode (i.e. inside the Cell processed by Cell.shard/F.shard).

    Note:
        - In the auto parallel mode, an exception will throw if the search mode is not
          "sharding_propagation".
        - In the semi-auto parallel mode, the parallel mode will automatically switch to auto
          parallel mode with the search mode be set to "sharding_propagation".
        - Currently, configuring multi-dimension and multi-copy reshard strategy in
          mindspore.Layout is not supported.

    Args:
        tensor (Tensor): The tensor to be set the sharding strategy.
        layout (Layout): The layout to shard the tensor precisely, including the device
                         arrangement (device_matrix) and the alias for the device matrix
                         (alias_name).

    Returns:
        Tensor. The mathematically equivalent of the input tensor.

    Raises:
        TypeError: Reshard takes in Tensor type as the first input param, but got: `type(tensor)`.
        TypeError: Reshard only support type mindspore.Layout but got: `type(layout)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import ops, nn, Tensor, context, Layout
        >>> context.set_context(mode=ms.GRAPH_MODE)
        >>> context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
        ...                                   search_mode="sharding_propagation")
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.matmul = ops.MatMul()
        ...         self.relu = ops.ReLU()
        ...     def construct(self, x, layout):
        ...         x = self.relu(x)
        ...         x_reshard = ops.reshard(x, layout)
        ...         y = Tensor(np.ones(shape=(128, 128)), dtype=ms.float32)
        ...         x = self.matmul(x_reshard, y)
        ...         return x
        >>>
        >>> layout = Layout((4, 2), ("dp", "mp"))
        >>> input_layout = layout("dp", "mp")
        >>> net = Network()
        >>> tensor = Tensor(np.ones(shape=(128, 128)), dtype=ms.float32)
        >>> out = net(tensor, input_layout)
    """
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Reshard takes in Tensor type as the first input param, but got: {type(tensor)}.")
    if not isinstance(layout, Layout):
        raise TypeError(f"Reshard only support type mindspore.Layout, but got: {type(layout)}.")

    def layout_to_tuple(layout):
        layout_dict = layout.to_dict()
        tensor_map = layout_dict["tensor_map"]
        device_matrix_rev = layout_dict["device_matrix"][::-1]
        axis_stgy = ()
        for ind in tensor_map:
            if ind == -1:
                axis_stgy += (1,)
            else:
                axis_stgy += (device_matrix_rev[ind],)
        return axis_stgy

    in_strategy = layout_to_tuple(layout)
    _reshard = _get_cache_prim(P.Reshard)(in_layout=(layout,), out_layout=(layout,), in_strategy=(in_strategy,))
    return _reshard(tensor)


def _redistribute(tensor, dst_dtensor_info):
    """
    Redistribute the tensor from the source sharding strategy to the destination sharding strategy.

    Args:
        tensor (Tensor): The source tensor.
        dst_dtensor_info (_DistributedTensorInfo): The destination sharding strategy.

    Returns:
        Tensor, value is same as the source tensor, but the sharding strategy is the destination sharding strategy.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import numpy as np
        >>> from mindspore.communication import init
        >>> from mindspore import Tensor, Layout, _DistributedTensorInfo
        >>>
        >>> init()
        >>> layout = Layout((2, 1), ("dp", "mp"))
        >>> src_layout = layout("dp", "mp")
        >>> distributed_info = _DistributedTensorInfo(src_layout)
        >>> x = Tensor(np.ones([2, 2]).astype(np.float32))
        >>> out = x.redistribute(distributed_info)
        >>> print(out)
        [[1. 1.]]
    """
    from mindspore.parallel._cell_wrapper import RedistributionCell, _insert_virtual_pp_dim
    if not isinstance(dst_dtensor_info, _DistributedTensorInfo):
        raise TypeError(
            "dst_dtensor_info should be _DistributedTensorInfo type, but got {}".format(type(dst_dtensor_info)))
    run_mode = context.get_context("mode")
    context.set_context(mode=context.GRAPH_MODE)
    og_auto_parallel_context, pp_config = _get_all_auto_parallel_context()
    context.reset_auto_parallel_context()
    tensor_data = tensor
    all_reduce_data = False
    # If src_pp_stages is less than or equal to dst_pp_stages, the parameters of each pp stage of src can be
    # directly swapped to the corresponding card of dst
    # rank0 01 11           01
    # rank1 02 12           02
    #  pp1   ------>  pp2
    # rank2 03 13           11
    # rank3 04 14           12
    # if dtensor info is None, return the all 1 strategy as from dtensor info
    if tensor._dtensor_info is None:
        all_dev_num = get_group_size()
        dev_mat = Layout((all_dev_num,), ("replica",))
        tensor_map = ["None"] * len(tensor.shape)
        layout = dev_mat(*tensor_map)
        tensor._dtensor_info = _DistributedTensorInfo(layout)
    src_layout_info = tensor._dtensor_info.layout.to_dict()
    dst_layout_info = dst_dtensor_info.layout.to_dict()
    if len(tensor._dtensor_info.layout.to_dict()["rank_list"]) < len(dst_dtensor_info.layout.to_dict()["rank_list"]):
        # If src_pp_stages is greater than dst_pp_stages, the weights of the corresponding cards need to
        # be communicated via AllReduce to swap. Need to communicate src rank0's 01 to src rank2,
        # so that rank2 holds param0's data. Similarly, communicate rank1's 02 to rank3
        # rank0 01           01 11
        # rank1 02           02 12
        # pp2 ------->  pp1
        # rank2 11           03 13
        # rank3 12           04 14
        from mindspore.parallel._cell_wrapper import CommTensorDataForPP
        if get_rank() in dst_dtensor_info.layout.to_dict()["rank_list"]:
            comm_tensor_data_func = CommTensorDataForPP(tensor_data, tensor._dtensor_info, dst_dtensor_info)
            tensor_data = comm_tensor_data_func.comm_data()
            all_reduce_data = True
    ms.communication.comm_func.barrier()
    dataset_strategy = (_insert_virtual_pp_dim(tensor._dtensor_info.layout),)
    if get_rank() not in tensor._dtensor_info.layout.to_dict()["rank_list"] and not all_reduce_data:
        dataset_strategy = "full_batch"
    context.set_auto_parallel_context(dataset_strategy=dataset_strategy,
                                      parallel_mode="semi_auto_parallel", device_num=get_group_size())
    global REDIST_CELL_CACHE
    redist_cache_key = (f"{src_layout_info['device_matrix']}, {src_layout_info['tensor_map']} -> "
                        f"{dst_layout_info['device_matrix']}, {dst_layout_info['tensor_map']}")
    if redist_cache_key in REDIST_CELL_CACHE.keys():
        logger.debug(f"redist_cache_key is {redist_cache_key}, match cache")
        redist_func = REDIST_CELL_CACHE[redist_cache_key]
    else:
        logger.debug(f"redist_cache_key is {redist_cache_key}, not match cache")
        redist_func = RedistributionCell(tensor._dtensor_info.layout, dst_dtensor_info.layout)
        REDIST_CELL_CACHE[redist_cache_key] = redist_func
    redist_func.set_train(True)
    redist_tensor_data = redist_func(tensor_data)
    context.reset_auto_parallel_context()
    _recover_auto_parallel_context(og_auto_parallel_context, pp_config)
    context.set_context(mode=run_mode)
    redist_tensor_data._dtensor_info = dst_dtensor_info
    return redist_tensor_data


__all__ = [
    'reshard'
]

__all__.sort()
