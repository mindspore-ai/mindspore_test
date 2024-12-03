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
import hashlib
import mindspore as ms
from mindspore import context
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    get_backend,
    new_group,
    get_global_rank,
    get_process_group_ranks,
    broadcast,
    gather,
    scatter,
    all_gather,
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_comm_cpu.py
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


def test_cpu_new_group():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    group = new_group(list(range(size)))
    name = "hccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    assert group == name
    group = new_group(list(range(size)), backend="hccl")
    assert group == name
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    group = new_group(list(range(size)), backend="mccl")
    assert group == name

    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name


def test_cpu_get_rank():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    local_rank = get_rank()
    assert local_rank == rank
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        local_rank = get_rank(group)
        global_rank = get_global_rank(group, local_rank)
        assert global_rank == rank
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        local_rank = get_rank(group)
        global_rank = get_global_rank(group, local_rank)
        assert global_rank == rank


def test_cpu_get_backend():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    backend = get_backend()
    assert backend == "hccl"
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        backend = get_backend(group)
        assert backend == "mccl"
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        backend = get_backend(group)
        assert backend == "mccl"


def test_cpu_get_process_group_ranks():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ranks = get_process_group_ranks()
    print("ranks is:", ranks)
    assert ranks == list(range(size))
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        ranks = get_process_group_ranks(group)
        assert ranks == list(range(2))
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        ranks = get_process_group_ranks(group)
        assert ranks == [2, 3]


def test_cpu_broadcast():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 0:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=0, group=group)
    assert output_handle is None
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    # 异步场景
    tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    if rank != 1:
        tensor = ms.Tensor(np.zeros([2, 4]).astype(np.float32))
    output_handle = broadcast(tensor, src=1, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    except_output_tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
    assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())

    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
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
    # group场景
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        tensor = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.int32))
        if rank != 2:
            tensor = ms.Tensor(np.zeros([2, 4]).astype(np.int32))
        output_handle = broadcast(tensor, src=2, group=name)
        assert output_handle is None
        except_output_tensor = ms.Tensor(
            np.arange(8).reshape([2, 4]).astype(np.int32)
        )
        assert np.allclose(tensor.asnumpy(), except_output_tensor.asnumpy())


def test_cpu_all_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
    # 同步场景
    input_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
    output_tensor = []
    except_output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
        except_output_tensor.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))

    output_handle = all_gather(output_tensor, input_tensor, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = all_gather(output_tensor, input_tensor, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
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
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.int64))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.int64)))
            except_output_tensor.append(input_tensor1)
        output_handle = all_gather(output_tensor1, input_tensor1, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )


def test_cpu_gather():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
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
    output_handle = gather(input_tensor, output_tensor, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # 异步场景
    output_tensor = []
    for _ in range(size):
        output_tensor.append(ms.Tensor(np.zeros([3, 3]).astype(np.float32)))
    output_handle = gather(input_tensor, output_tensor, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor[0].asnumpy(), except_output_tensor[0].asnumpy())
    assert np.allclose(output_tensor[1].asnumpy(), except_output_tensor[1].asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
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
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float16))
        output_tensor1 = []
        except_output_tensor = []
        for _ in range(2):
            output_tensor1.append(ms.Tensor(np.zeros([3, 3]).astype(np.float16)))
            if rank == 3:
                except_output_tensor.append(input_tensor1)
            else:
                except_output_tensor.append(
                    ms.Tensor(np.zeros([3, 3]).astype(np.float16))
                )
        output_handle = gather(input_tensor1, output_tensor1, dst=3, group=name)
        assert output_handle is None
        assert np.allclose(
            output_tensor1[0].asnumpy(), except_output_tensor[0].asnumpy()
        )
        assert np.allclose(
            output_tensor1[1].asnumpy(), except_output_tensor[1].asnumpy()
        )


def test_cpu_scatter():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    name = "mccl_" + str(size) + "_" + hashlib.sha1(bytes("_".join(map(str, range(size))), "utf-8")).hexdigest()
    group = new_group(list(range(size)), backend="mccl")
    assert group == name
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
    output_handle = scatter(output_tensor, input_tensor, src=0, group=group)
    assert output_handle is None
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # 异步场景
    output_tensor = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
    output_handle = scatter(output_tensor, input_tensor, src=0, group=group, async_op=True)
    assert output_handle is not None
    output_handle.wait()
    assert np.allclose(output_tensor.asnumpy(), except_output_tensor.asnumpy())
    # group场景
    if rank == 0 or rank == 1:
        group = new_group(list(range(2)), backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, range(2))), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.float32)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.float32))
        output_handle = scatter(output_tensor1, input_tensor1, src=0, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.float32))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
    if rank == 2 or rank == 3:
        group = new_group([2, 3], backend="mccl")
        name = "mccl_" + str(2) + "_" + hashlib.sha1(bytes("_".join(map(str, [2, 3])), "utf-8")).hexdigest()
        assert group == name
        input_tensor1 = []
        for _ in range(2):
            input_tensor1.append(ms.Tensor(np.ones([3, 3]).astype(np.int8)))
        output_tensor1 = ms.Tensor(np.zeros([3, 3]).astype(np.int8))
        output_handle = scatter(output_tensor1, input_tensor1, src=2, group=name)
        except_output_tensor = ms.Tensor(np.ones([3, 3]).astype(np.int8))
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())
