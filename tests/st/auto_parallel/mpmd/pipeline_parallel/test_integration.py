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
import numpy as np
from mindspore import Tensor, dtype as mstype
from mindspore.parallel import Layout
from mindspore.parallel.mpmd.pipeline_parallel._utils import _MicroBatch, BatchDimSpec


def test_microbatch_basic_args_only():
    """
    Description:Test the function of args
    Expectation:Run success
    """


    batch_size = 8
    micro_batch_num = 4
    input_tensor1 = Tensor(np.ones((batch_size, 10, 20)), mstype.float32)
    input_tensor2 = Tensor(np.ones((batch_size, 5)), mstype.float32)

    dp = 1
    mp = 1
    cp = 1
    layout = Layout((dp, mp, cp), ("dp", "mp", "cp"))
    input_tensor1_layout = layout("dp", "mp", "cp")
    input_tensor2_layout = layout("dp", "mp")
    input_tensor1.local_to_global(input_tensor1_layout)
    input_tensor2.local_to_global(input_tensor2_layout)

    original_layout1 = input_tensor1.layout
    original_layout2 = input_tensor2.layout

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=BatchDimSpec.from_tuple((0, 0))
    )

    args = [input_tensor1, input_tensor2]
    kwargs = {}
    args_after_split, kwargs_after_split = microbatch(args, kwargs)

    assert len(args_after_split) == micro_batch_num
    assert len(kwargs_after_split) == micro_batch_num

    for i, micro_args in enumerate(args_after_split):
        assert micro_args[0].shape == (batch_size // micro_batch_num, 10, 20)
        assert micro_args[1].shape == (batch_size // micro_batch_num, 5)

        assert micro_args[0].layout == original_layout1, f"Micro batch {i} tensor1 layout mismatch"
        assert micro_args[1].layout == original_layout2, f"Micro batch {i} tensor2 layout mismatch"

    for micro_kwargs in kwargs_after_split:
        assert micro_kwargs == {}


def test_microbatch_kwargs_only():
    """
    Description:Test the function of kwargs
    Expectation:Run success
    """


    batch_size = 12
    micro_batch_num = 3
    input_tensor1 = Tensor(np.ones((batch_size, 32)), mstype.float32)
    input_tensor2 = Tensor(np.ones((20, batch_size)), mstype.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    input_tensor1.local_to_global(input_tensor_layout)
    input_tensor2.local_to_global(input_tensor_layout)

    original_layout1 = input_tensor1.layout
    original_layout2 = input_tensor2.layout

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        kwargs_batch_dim=BatchDimSpec.from_dict({
            "input1": 0,
            "input2": 1
        })
    )

    args = []
    kwargs = {
        "input1": input_tensor1,
        "input2": input_tensor2
    }
    args_after_split, kwargs_after_split = microbatch(args, kwargs)

    assert len(args_after_split) == micro_batch_num
    assert len(kwargs_after_split) == micro_batch_num

    for i, micro_kwargs in enumerate(kwargs_after_split):
        assert micro_kwargs['input1'].shape == (batch_size // micro_batch_num, 32)
        assert micro_kwargs['input2'].shape == (20, batch_size // micro_batch_num)

        assert micro_kwargs['input1'].layout == original_layout1, f"Micro batch {i} tensor1 layout mismatch"
        assert micro_kwargs['input2'].layout == original_layout2, f"Micro batch {i} tensor2 layout mismatch"


    for micro_args in args_after_split:
        assert micro_args == []

def test_microbatch_mixed_args_kwargs():
    """
    Description:Test the mix function of args and kwargs
    Expectation:Run success
    """


    batch_size = 16
    micro_batch_num = 2
    arg_tensor = Tensor(np.ones((batch_size, 64)), mstype.float32)
    kwarg_tensor1 = Tensor(np.ones((8, batch_size, 16)), mstype.float32)
    kwarg_tensor2 = Tensor(np.ones((batch_size,)), mstype.float32)

    dp = 1
    mp = 1
    cp = 1
    layout = Layout((dp, mp, cp), ("dp", "mp", "cp"))
    arg_tensor_layout = layout("dp", "mp")
    kwarg_tensor1_layout = layout("dp", "mp", "cp")
    kwarg_tensor2_layout = layout("dp")
    arg_tensor.local_to_global(arg_tensor_layout)
    kwarg_tensor1.local_to_global(kwarg_tensor1_layout)
    kwarg_tensor2.local_to_global(kwarg_tensor2_layout)

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=BatchDimSpec.from_tuple((0,)),
        kwargs_batch_dim=BatchDimSpec.from_dict({
            "data1": 1,
            "data2": 0
        })
    )

    args = [arg_tensor]
    kwargs = {
        "data1": kwarg_tensor1,
        "data2": kwarg_tensor2
    }
    args_after_split, kwargs_after_split = microbatch(args, kwargs)

    assert len(args_after_split) == micro_batch_num
    assert len(kwargs_after_split) == micro_batch_num

    for micro_args, micro_kwargs in zip(args_after_split, kwargs_after_split):
        assert len(micro_args) == 1
        assert micro_args[0].shape == (batch_size // micro_batch_num, 64)

        assert micro_kwargs['data1'].shape == (8, batch_size // micro_batch_num, 16)
        assert micro_kwargs['data2'].shape == (batch_size // micro_batch_num,)


def test_microbatch_default_batch_dim():
    """
    Description:Test the function of default_batch_dim
    Expectation:Run success
    """


    batch_size = 6
    micro_batch_num = 3
    input_tensor = Tensor(np.ones((batch_size, 10)), mstype.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    input_tensor.local_to_global(input_tensor_layout)

    microbatch = _MicroBatch(micro_batch_num=micro_batch_num)

    args = [input_tensor]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    for micro_args in args_after_split:
        assert micro_args[0].shape == (batch_size // micro_batch_num, 10)


def test_microbatch_partial_batch_dim_spec():
    """
    Description:Test the function of partial_batch_dim
    Expectation:Run success
    """


    batch_size = 9
    micro_batch_num = 3
    arg_tensor1 = Tensor(np.ones((batch_size, 10)), mstype.float32)
    arg_tensor2 = Tensor(np.ones((batch_size, 20)), mstype.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    arg_tensor1.local_to_global(input_tensor_layout)
    arg_tensor2.local_to_global(input_tensor_layout)

    microbatch = _MicroBatch(
        micro_batch_num=micro_batch_num,
        args_batch_dim=[BatchDimSpec(1), None]
    )  # BatchDimSpec.from_tuple((1,)):IndexError: tuple index out of range

    args = [arg_tensor1, arg_tensor2]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    for micro_args in args_after_split:
        assert len(micro_args) == 2


def test_microbatch_edge_cases():
    """
    Description:Test the function of edge_cases
    Expectation:Run success
    """


    batch_size = 8
    input_tensor = Tensor(np.ones((batch_size, 10)), mstype.float32)

    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"))
    input_tensor_layout = layout("dp", "mp")
    input_tensor.local_to_global(input_tensor_layout)

    microbatch = _MicroBatch(micro_batch_num=1)
    args = [input_tensor]
    kwargs = {}
    args_after_split, _ = microbatch(args, kwargs)

    assert len(args_after_split) == 1
    assert args_after_split[0][0].shape == (batch_size, 10)

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)

    test_microbatch_basic_args_only()
    test_microbatch_kwargs_only()
    test_microbatch_mixed_args_kwargs()
    test_microbatch_default_batch_dim()
    test_microbatch_partial_batch_dim_spec()
    test_microbatch_edge_cases()
