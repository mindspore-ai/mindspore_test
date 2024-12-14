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
import numpy as np
import pytest
import hashlib
import mindspore as ms
from mindspore import context
from mindspore.common.api import _pynative_executor
from mindspore.ops import ReduceOp, cat
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    new_group,
    get_backend,
    get_global_rank,
    get_process_group_ranks,
    get_group_rank,
    all_reduce,
    all_gather_into_tensor,
    all_to_all,
    all_to_all_single,
    reduce_scatter_tensor,
    isend,
    irecv,
    send,
    recv,
    barrier,
    broadcast,
    reduce,
    P2POp,
    batch_isend_irecv,
    gather,
    scatter,
    all_gather,
    reduce_scatter,
)

#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True --cluster_time_out=800  pytest -sv --disable-warnings test_distributed.py
np.random.seed(1)
init_process_group()
context.set_auto_parallel_context(
    parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True
)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
rank = get_rank()
size = get_world_size()
if size % 2 != 0:
    raise RuntimeError("Group size should be divided by 2 exactly.")


def test_hccl_new_group():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    group = new_group()
    print("group is:", group)
    assert group == "hccl_world_group"
    group = new_group()
    assert group == "hccl_world_group"
    group = new_group(None)
    assert group == "hccl_world_group"
    group = new_group(list(range(size)))
    name = "hccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    assert group == name
    group = new_group(list(range(size)))
    assert group == name
    group = new_group(list(range(size)), 1)
    assert group == name
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
    #超时用例
    #group = new_group(list(range(9)))
    #assert group == ""
    with pytest.raises(TypeError):
        new_group(1)
    with pytest.raises(TypeError):
        new_group(True)
    if rank == 0 or rank == 1:
        with pytest.raises(ValueError):
            new_group([0, 0, 1, 1])
    if rank == 0 or rank == 1:
        group = new_group([2, 3])
        assert group == ""


def test_hccl_get_backend():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    backend = get_backend()
    assert backend == "hccl"
    backend = get_backend(None)
    assert backend == "hccl"
    name = "hccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), 1)
    assert group == name
    backend = get_backend(group)
    assert backend == "hccl"
    with pytest.raises(TypeError):
        backend = get_backend(1)


def test_hccl_get_global_rank():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    global_rank = get_global_rank(None, rank)
    assert global_rank == rank
    with pytest.raises(TypeError):
        get_global_rank(0, rank)
    with pytest.raises(TypeError):
        get_global_rank(None, "rank")
    if rank == 0:
        group = new_group(list(range(2)))
        global_rank = get_global_rank(group, 1)
        assert global_rank == 1
    if rank == 1:
        group = new_group(list(range(2)))
        global_rank = get_global_rank(group, 0)
        assert global_rank == 0
    if rank == 2:
        group = new_group([2, 3])
        global_rank = get_global_rank(group, 1)
        assert global_rank == 3
    if rank == 3:
        group = new_group([2, 3])
        global_rank = get_global_rank(group, 0)
        assert global_rank == 2


def test_hccl_get_group_rank():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    group_rank = get_group_rank(None, rank)
    assert group_rank == rank
    with pytest.raises(TypeError):
        get_group_rank(0, rank)
    with pytest.raises(TypeError):
        get_group_rank(None, "rank")
    if rank == 0:
        group = new_group(list(range(2)))
        group_rank = get_group_rank(group, 1)
        assert group_rank == 1
    if rank == 1:
        group = new_group(list(range(2)))
        group_rank = get_group_rank(group, 0)
        assert group_rank == 0
    if rank == 2:
        group = new_group([2, 3])
        global_rank = get_group_rank(group, 3)
        assert global_rank == 1
    if rank == 3:
        group = new_group([2, 3])
        global_rank = get_group_rank(group, 2)
        assert global_rank == 0


def test_hccl_get_process_group_ranks():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ranks = get_process_group_ranks()
    print("ranks is:", ranks)
    assert ranks == list(range(size))
    with pytest.raises(TypeError):
        get_process_group_ranks(0)
    with pytest.raises(TypeError):
        get_process_group_ranks(True)
    if rank == 0:
        group = new_group(list(range(2)))
        ranks = get_process_group_ranks(group)
        assert ranks == list(range(2))
    if rank == 1:
        group = new_group(list(range(2)))
        ranks = get_process_group_ranks(group)
        assert ranks == list(range(2))
    if rank == 2:
        group = new_group([2, 3])
        ranks = get_process_group_ranks(group)
        assert ranks == [2, 3]
    if rank == 3:
        group = new_group([2, 3])
        ranks = get_process_group_ranks(group)
        assert ranks == [2, 3]


def test_hccl_all_reduce_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))

    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor, op=ReduceOp.SUM)
    assert sum_output_handle is None
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))

    max_input_tensor = input_tensor * (rank + 1)
    max_output_handle = all_reduce(max_input_tensor, op=ReduceOp.MAX)
    assert max_output_handle is None
    except_max_output = input_tensor * size

    min_input_tensor = input_tensor * (rank + 1)
    min_output_handle = all_reduce(min_input_tensor, op=ReduceOp.MIN)
    assert min_output_handle is None
    except_min_output = input_tensor

    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    assert np.allclose(max_input_tensor.asnumpy(), except_max_output.asnumpy())
    assert np.allclose(min_input_tensor.asnumpy(), except_min_output.asnumpy())


def test_hccl_all_reduce():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    # 同步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor)
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    assert sum_output_handle is None
    # 异步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = all_reduce(sum_input_tensor, async_op=True)
    assert sum_output_handle is not None
    sum_output_handle.wait()
    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        sum_input_tensor1 = input_tensor * (rank + 1)
        sum_output_handle = all_reduce(sum_input_tensor1, group=name)
        except_sum_output = input_tensor * (sum(list(range(1, 3))))
        assert np.allclose(sum_input_tensor1.asnumpy(), except_sum_output.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_reduce(1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, op=1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, op="test")
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, group=1)
    with pytest.raises(TypeError):
        all_reduce(sum_input_tensor, async_op="test")


def test_hccl_all_gather_into_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    output_handle = all_gather_into_tensor(output_tensor, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.float32))
    output_handle = all_gather_into_tensor(output_tensor, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
        output_handle = all_gather_into_tensor(
            output_tensor1, input_tensor1, group=name
        )
        except_output_tensor = cat([input_tensor1, input_tensor1])
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_gather_into_tensor(1)
    with pytest.raises(TypeError):
        all_gather_into_tensor(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        all_gather_into_tensor(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        all_gather_into_tensor([1], input_tensor)
    with pytest.raises(TypeError):
        all_gather_into_tensor(output_tensor, [1])
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3 * size]).astype(np.float32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3 * size, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        all_gather_into_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()


def test_hccl_reduce_scatter_tensor_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    sum_input_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_sum_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    sum_output_handle = reduce_scatter_tensor(
        sum_input_tensor, input_tensor, op=ReduceOp.SUM
    )
    assert sum_output_handle is None
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32)) * (rank + 1)
    max_input_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_max_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    sum_output_handle = reduce_scatter_tensor(
        max_input_tensor, input_tensor, op=ReduceOp.MAX
    )
    assert sum_output_handle is None
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32)) * (rank + 1)
    min_input_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_min_output = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * 1
    sum_output_handle = reduce_scatter_tensor(
        min_input_tensor, input_tensor, op=ReduceOp.MIN
    )
    assert sum_output_handle is None
    assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    assert np.allclose(max_input_tensor.asnumpy(), except_max_output.asnumpy())
    assert np.allclose(min_input_tensor.asnumpy(), except_min_output.asnumpy())


def test_hccl_reduce_scatter_tensor():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3 * size, 3]).astype(np.float32))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    output_handle = reduce_scatter_tensor(output_tensor, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = reduce_scatter_tensor(output_tensor, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.ones([3 * 2, 3]).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = reduce_scatter_tensor(output_tensor1, input_tensor1, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * 2
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        reduce_scatter_tensor(1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(output_tensor, input_tensor, op=1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(output_tensor, input_tensor, op="test")
    with pytest.raises(TypeError):
        reduce_scatter_tensor(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        reduce_scatter_tensor([1], input_tensor)
    with pytest.raises(TypeError):
        reduce_scatter_tensor(output_tensor, [1])
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(ValueError):
        reduce_scatter_tensor(output_tensor, input_tensor)
        _pynative_executor.sync()


def test_hccl_reduce_type():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))

    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = reduce(sum_input_tensor, dst=0, op=ReduceOp.SUM)
    assert sum_output_handle is None
    except_sum_output = input_tensor * (sum(list(range(1, size + 1))))

    max_input_tensor = input_tensor * (rank + 1)
    max_output_handle = reduce(max_input_tensor, dst=0, op=ReduceOp.MAX)
    assert max_output_handle is None
    except_max_output = input_tensor * size

    min_input_tensor = input_tensor * (rank + 1)
    min_output_handle = reduce(min_input_tensor, dst=0, op=ReduceOp.MIN)
    assert min_output_handle is None
    except_min_output = input_tensor
    if rank == 0:
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
        assert np.allclose(max_input_tensor.asnumpy(), except_max_output.asnumpy())
        assert np.allclose(min_input_tensor.asnumpy(), except_min_output.asnumpy())
    else:
        except_output = input_tensor * (rank + 1)
        assert np.allclose(sum_input_tensor.asnumpy(), except_output.asnumpy())
        assert np.allclose(max_input_tensor.asnumpy(), except_output.asnumpy())
        assert np.allclose(min_input_tensor.asnumpy(), except_output.asnumpy())


def test_hccl_reduce():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    input_tensor = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
    # 同步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = reduce(sum_input_tensor, dst=0)
    assert sum_output_handle is None
    if rank == 0:
        except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    else:
        except_sum_output = input_tensor * (rank + 1)
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())

    # 异步场景
    sum_input_tensor = input_tensor * (rank + 1)
    sum_output_handle = reduce(sum_input_tensor, dst=1, async_op=True)
    assert sum_output_handle is not None
    sum_output_handle.wait()
    if rank == 1:
        except_sum_output = input_tensor * (sum(list(range(1, size + 1))))
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    else:
        except_sum_output = input_tensor * (rank + 1)
        assert np.allclose(sum_input_tensor.asnumpy(), except_sum_output.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        sum_input_tensor1 = input_tensor * (rank + 1)
        sum_output_handle = reduce(sum_input_tensor1, dst=1, group=name)
        if rank == 1:
            except_sum_output = input_tensor * (sum(list(range(1, 2 + 1))))
            assert np.allclose(sum_input_tensor1.asnumpy(), except_sum_output.asnumpy())
        else:
            except_sum_output = input_tensor * (rank + 1)
            assert np.allclose(sum_input_tensor1.asnumpy(), except_sum_output.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        reduce(1)
    with pytest.raises(TypeError):
        reduce(sum_input_tensor, dst="test")
    with pytest.raises(TypeError):
        reduce(sum_input_tensor, dst=0, op=1)
    with pytest.raises(TypeError):
        reduce(sum_input_tensor, dst=0, op="test")
    with pytest.raises(TypeError):
        reduce(sum_input_tensor, dst=0, group=1)
    with pytest.raises(TypeError):
        reduce(sum_input_tensor, dst=0, async_op="test")


def test_hccl_batch_isend_irecv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    barrier()
    # 异步场景
    if rank == 0 or rank == 1:
        next_rank = (rank + 1) % 2
        prev_rank = (rank + size - 1) % 2

        send_tensor = ms.Tensor(rank + 1, dtype=ms.float32)
        recv_tensor = ms.Tensor(0.0, dtype=ms.float32)

        send_op = P2POp("isend", send_tensor, next_rank)
        recv_op = P2POp("irecv", recv_tensor, prev_rank)

        p2p_op_list = [send_op, recv_op]
        output = batch_isend_irecv(p2p_op_list)
        assert len(output) == 1
        assert output[0] is not None
        output[0].wait()
        if rank == 1:
            except_output = ms.Tensor(1, dtype=ms.float32)
            assert np.allclose(recv_tensor.asnumpy(), except_output.asnumpy())
        else:
            except_output = ms.Tensor(2, dtype=ms.float32)
            assert np.allclose(recv_tensor.asnumpy(), except_output.asnumpy())
        # 异常场景
        send_op = P2POp("isend", send_tensor, next_rank)
        recv_op = P2POp("irecv", recv_tensor, prev_rank, group="11")
        with pytest.raises(TypeError):
            batch_isend_irecv()

    if rank == 0 or rank == 1:
        next_rank = (rank + 1) % 2
        prev_rank = (rank + size - 1) % 2

        send_tensor = ms.Tensor(rank + 1, dtype=ms.float32)
        recv_tensor = ms.Tensor(0.0, dtype=ms.float32)

        send_op = P2POp(isend, send_tensor, next_rank)
        recv_op = P2POp(irecv, recv_tensor, prev_rank)

        p2p_op_list = [send_op, recv_op]
        output = batch_isend_irecv(p2p_op_list)
        assert len(output) == 1
        assert output[0] is not None
        output[0].wait()
        if rank == 1:
            except_output = ms.Tensor(1, dtype=ms.float32)
            assert np.allclose(recv_tensor.asnumpy(), except_output.asnumpy())
        else:
            except_output = ms.Tensor(2, dtype=ms.float32)
            assert np.allclose(recv_tensor.asnumpy(), except_output.asnumpy())
        # 异常场景
        send_op = P2POp(isend, send_tensor, next_rank)
        recv_op = P2POp(irecv, recv_tensor, prev_rank, group="11")
        with pytest.raises(TypeError):
            batch_isend_irecv()

    # 异常场景
    with pytest.raises(TypeError):
        batch_isend_irecv()
    with pytest.raises(TypeError):
        batch_isend_irecv(1)
    with pytest.raises(TypeError):
        batch_isend_irecv([])
    with pytest.raises(TypeError):
        batch_isend_irecv([1])
    barrier()


def test_hccl_broadcast():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 0:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=0)
    assert output_handle is None
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    # 异步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 1:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=1, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
        if rank != 0:
            tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
        output_handle = broadcast(tensor, src=0, group=name)
        assert output_handle is None
        except_output_tensor = ms.Tensor(
            np.arange(8).reshape([2, 4]).astype(np.float32)
        )
        assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        broadcast(1, src=0)
    with pytest.raises(TypeError):
        broadcast(tensor, src="test")
    with pytest.raises(TypeError):
        broadcast(tensor, src=0, group=1)
    with pytest.raises(TypeError):
        broadcast(tensor, src=0, async_op="test")


def test_hccl_barrier():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    output_handle = barrier()
    assert output_handle is None
    # 异步场景
    output_handle = barrier(async_op=True)
    assert output_handle is not None
    output_handle.wait()
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        output_handle = barrier(group=name)
        assert output_handle is None
    # 异常场景
    with pytest.raises(TypeError):
        barrier(group=1)
    with pytest.raises(TypeError):
        barrier(async_op="test")


def test_hccl_send():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        out = recv(output, src=rank - 1)
        assert out == 0
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out = recv(output, src=1, group=group)
            assert out == 0
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())

    # 异常场景
    with pytest.raises(TypeError):
        send(1)
    with pytest.raises(TypeError):
        send(input_tensor, dst="test")
    with pytest.raises(TypeError):
        send(input_tensor, group=1)
    with pytest.raises(ValueError):
        send(input_tensor, dst=rank)


def test_hccl_recv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        out = recv(output, src=rank - 1)
        assert out == 0
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            out = recv(output, src=1, group=group)
            assert out == 0
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        recv(1)
    with pytest.raises(TypeError):
        recv(output, src="test")
    with pytest.raises(TypeError):
        recv(output, group=1)


def test_hccl_isend():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 异步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        handle = isend(input_tensor, rank + 1 % size)
        assert handle is not None
        handle.wait()
    else:
        out = recv(output, src=rank - 1)
        assert out == 0
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            handle = isend(input_tensor, dst=0, group=group)
            assert handle is not None
            handle.wait()
        else:
            out = recv(output, src=1, group=group)
            assert out == 0
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())

    # 异常场景
    with pytest.raises(TypeError):
        isend(1)
    with pytest.raises(TypeError):
        isend(input_tensor, dst="test")
    with pytest.raises(TypeError):
        isend(input_tensor, group=1)
    with pytest.raises(ValueError):
        isend(input_tensor, dst=rank)


def test_hccl_irecv():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 异步场景
    input_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    output = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    if rank % 2 == 0:
        send(input_tensor, rank + 1 % size)
    else:
        handle = irecv(output, src=rank - 1)
        assert handle is not None
        handle.wait()
        assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 1:
            send(input_tensor, dst=0, group=group)
        else:
            handle = irecv(output, src=1, group=group)
            assert handle is not None
            handle.wait()
            assert np.allclose(output.asnumpy(), input_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        irecv(1)
    with pytest.raises(TypeError):
        irecv(output, src="test")
    with pytest.raises(TypeError):
        irecv(output, group=1)


def test_hccl_all_to_all():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([1, 1]).astype(np.float32)) * rank
    input_tensors = []
    output_tensors = []
    except_output_tensors = []
    for i in range(size):
        input_tensors.append(input_tensor)
        output_tensors.append(ms.Tensor(np.zeros([1, 1]).astype(np.float32)))
        except_output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.float32)) * i)

    handle = all_to_all(output_tensors, input_tensors)
    assert handle is None
    assert np.allclose(output_tensors[0].asnumpy(), except_output_tensors[0].asnumpy())
    assert np.allclose(output_tensors[1].asnumpy(), except_output_tensors[1].asnumpy())
    # 异步场景

    except_output_tensors = []
    for i in range(size):
        output_tensors.append(ms.Tensor(np.zeros([1, 1]).astype(np.float32)))
        except_output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.float32)) * i)

    handle = all_to_all(output_tensors, input_tensors, async_op=True)
    assert handle is not None
    handle.wait()
    assert np.allclose(output_tensors[0].asnumpy(), except_output_tensors[0].asnumpy())
    assert np.allclose(output_tensors[1].asnumpy(), except_output_tensors[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 0:
            send_tensor_list = [ms.Tensor(1.0), ms.Tensor([[2, 3], [4, 5.0]])]
            recv_tensor_list = [ms.Tensor((0), dtype=ms.float32), ms.Tensor([0, 0.0])]
            handle = all_to_all(recv_tensor_list, send_tensor_list, group=group)
            assert handle is None
            except_output_tensor = [
                ms.Tensor((1), dtype=ms.float32),
                ms.Tensor([2, 2.0]),
            ]
            assert np.allclose(
                recv_tensor_list[0].asnumpy(), except_output_tensor[0].asnumpy()
            )
            assert np.allclose(
                recv_tensor_list[1].asnumpy(), except_output_tensor[1].asnumpy()
            )
        if rank == 1:
            send_tensor_list = [ms.Tensor([2, 2.0]), ms.Tensor([4, 5, 6, 7.0])]
            recv_tensor_list = [
                ms.Tensor([[0, 0.0], [0, 0]]),
                ms.Tensor([0, 0, 0, 0.0]),
            ]
            handle = all_to_all(recv_tensor_list, send_tensor_list, group=group)
            assert handle is None
            except_output_tensor = [
                ms.Tensor([[2, 3.0], [4, 5]]),
                ms.Tensor([4, 5, 6, 7.0]),
            ]
            assert np.allclose(
                recv_tensor_list[0].asnumpy(), except_output_tensor[0].asnumpy()
            )
            assert np.allclose(
                recv_tensor_list[1].asnumpy(), except_output_tensor[1].asnumpy()
            )
    # 异常场景
    with pytest.raises(TypeError):
        all_to_all(1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, 1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, input_tensors, group=1)
    with pytest.raises(TypeError):
        all_to_all(output_tensors, input_tensors, async_op="1")
    with pytest.raises(ValueError):
        output_tensors = []
        for _ in range(size):
            output_tensors.append(ms.Tensor(np.ones([1, 1]).astype(np.int32)))
        all_to_all(output_tensors, input_tensors)
        _pynative_executor.sync()


def test_hccl_all_to_all_single():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([size, 1]).astype(np.float32)) * rank
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    handle = all_to_all_single(output_tensor, input_tensor)
    assert handle is None
    except_output_tensor = ms.Tensor(
        np.arange(size).reshape([size, 1]).astype(np.float32)
    )
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.float32))
    handle = all_to_all_single(output_tensor, input_tensor, async_op=True)
    assert handle is not None
    handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        if rank == 0:
            tensor = ms.Tensor([[0, 1.0, 2.0], [3, 4, 5], [6, 7, 8], [0, 0, 0]])
            output = ms.Tensor(np.zeros([4, 3]).astype(np.float32))
            handle = all_to_all_single(output, tensor, [3, 1], [3, 1], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor(
                [[0, 1.0, 2.0], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
            )
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
        if rank == 1:
            tensor = ms.Tensor([[9, 10.0, 11], [12.0, 13, 14], [1, 1, 1]])
            output = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
            handle = all_to_all_single(output, tensor, [1, 2], [1, 2], group=group)
            assert handle is None
            except_output_tensor = ms.Tensor([[0, 0, 0.0], [12, 13, 14], [1, 1, 1]])
            assert np.allclose(output.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        all_to_all_single(1, input_tensor)
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, 1)
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        all_to_all_single(output_tensor, input_tensor, async_op="1")
    with pytest.raises(ValueError):
        input_tensor = ms.Tensor(np.ones([size - 1, 1]).astype(np.float32))
        all_to_all_single(output_tensor, input_tensor)
    with pytest.raises(ValueError):
        input_tensor = ms.Tensor(np.ones([size, 1]).astype(np.float32)) * rank
        output_tensor = ms.Tensor(np.zeros([size, 1]).astype(np.int32))
        all_to_all_single(output_tensor, input_tensor)
        _pynative_executor.sync()


def test_hccl_all_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = []
    except_output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
        except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))

    output_handle = all_gather(output_tensor, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = all_gather(output_tensor, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
            except_output_tensor.append(input_tensor1)
        output_handle = all_gather(output_tensor1, input_tensor1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )
    # 异常场景
    with pytest.raises(TypeError):
        all_gather(1)
    with pytest.raises(TypeError):
        all_gather(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        all_gather(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        all_gather([1], input_tensor)
    with pytest.raises(TypeError):
        all_gather(output_tensor, [1])
    with pytest.raises(TypeError):
        output_tensor = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([3, 3]).astype(np.int32)),
        ]
        all_gather(output_tensor, input_tensor)
    with pytest.raises(TypeError):
        output_tensor = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([1, 3]).astype(np.float32)),
        ]
        all_gather(output_tensor, input_tensor)
    with pytest.raises(TypeError):
        output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.int32)))
        all_gather(output_tensor, input_tensor)
        _pynative_executor.sync()


def test_hccl_reduce_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = []
    for _ in range(size):
        input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * size
    output_handle = reduce_scatter(output_tensor, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = reduce_scatter(output_tensor, input_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = reduce_scatter(output_tensor1, input_tensor1, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32)) * 2
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        reduce_scatter(1)
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor, op=1)
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor, op="test")
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        reduce_scatter([1], input_tensor)
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, [1])
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([3, 3]).astype(np.int32)),
        ]
        reduce_scatter(output_tensor, input_tensor1)
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([1, 3]).astype(np.float32)),
        ]
        reduce_scatter(output_tensor, input_tensor1)
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(TypeError):
        reduce_scatter(output_tensor, input_tensor)
        _pynative_executor.sync()


def test_hccl_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = []
    except_output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
        if rank == 0:
            except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        else:
            except_output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = gather(input_tensor, output_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = gather(input_tensor, output_tensor, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
            if rank == 1:
                except_output_tensor.append(input_tensor1)
            else:
                except_output_tensor.append(
                    ms.Tensor(np.zeros([3, 3]).astype(np.float32))
                )
        output_handle = gather(input_tensor1, output_tensor1, dst=1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )
    # 异常场景
    with pytest.raises(TypeError):
        gather(1)
    with pytest.raises(TypeError):
        gather(input_tensor, output_tensor, group=1)
    with pytest.raises(TypeError):
        gather(input_tensor, output_tensor, dst="test")
    with pytest.raises(TypeError):
        gather(input_tensor, output_tensor, async_op="test")
    with pytest.raises(TypeError):
        gather([1], output_tensor)
    with pytest.raises(TypeError):
        gather(input_tensor, [1])
    with pytest.raises(TypeError):
        output_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([3, 3]).astype(np.int32)),
        ]
        gather(input_tensor, output_tensor1)
    with pytest.raises(TypeError):
        output_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([1, 3]).astype(np.float32)),
        ]
        gather(input_tensor, output_tensor1)
    with pytest.raises(TypeError):
        output_tensor = []
        for _ in range(size):
            output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.int32)))
        gather(input_tensor, output_tensor, dst=rank)
        _pynative_executor.sync()


def test_hccl_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # 同步场景
    input_tensor = []
    for _ in range(size):
        input_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
    if rank != 0:
        input_tensor = []
        for _ in range(size):
            input_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_handle = scatter(output_tensor, input_tensor, src=0)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = scatter(output_tensor, input_tensor, src=0, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), 1)
        name = "hccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = scatter(output_tensor1, input_tensor1, src=0, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    # 异常场景
    with pytest.raises(TypeError):
        scatter(1)
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src="test")
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, group=1)
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, async_op="test")
    with pytest.raises(TypeError):
        scatter([1], input_tensor)
    with pytest.raises(TypeError):
        scatter(output_tensor, [1])
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([3, 3]).astype(np.int32)),
        ]
        scatter(output_tensor, input_tensor1)
        _pynative_executor.sync()
    with pytest.raises(TypeError):
        input_tensor1 = [
            ms.Tensor(np.zeros([3, 3]).astype(np.float32)),
            ms.Tensor(np.zeros([1, 3]).astype(np.float32)),
        ]
        scatter(output_tensor, input_tensor1)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 1]).astype(np.float32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.int32))
    with pytest.raises(TypeError):
        scatter(output_tensor, input_tensor, src=rank)
        _pynative_executor.sync()


def test_hccl_scalar():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    # gather场景
    input_tensor = ms.Tensor(1)
    output_gather = []
    output_all_gather = []
    except_output_gather = []
    except_output_all_gather = []
    for _ in range(size):
        output_gather.append(ms.Tensor(0))
        output_all_gather.append(ms.Tensor(0))
        except_output_all_gather.append(ms.Tensor(1))
        if rank == 0:
            except_output_gather.append(ms.Tensor(1))
        else:
            except_output_gather.append(ms.Tensor(0))
    output_handle = gather(input_tensor, output_gather)
    assert output_handle is None
    assert np.allclose(output_gather[0].asnumpy(), except_output_gather[0].asnumpy())
    assert np.allclose(output_gather[1].asnumpy(), except_output_gather[1].asnumpy())

    output_handle = all_gather(output_all_gather, input_tensor)
    assert output_handle is None
    assert np.allclose(output_all_gather[0].asnumpy(), except_output_all_gather[0].asnumpy())
    assert np.allclose(output_all_gather[1].asnumpy(), except_output_all_gather[1].asnumpy())

    output_tensor = ms.Tensor(np.zeros([size]).astype(np.int64))
    except_output_tensor = ms.Tensor(np.ones([size]).astype(np.int64))
    output_handle = all_gather_into_tensor(output_tensor, input_tensor)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
