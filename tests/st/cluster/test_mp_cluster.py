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
"""
This test is for multiprocess launch.
"""
import os
import numpy as np
import mindspore.multiprocessing as mp
import mindspore as ms
from mindspore import context
from mindspore.ops import cat
from mindspore.mint.distributed import init_process_group, all_gather_into_tensor
from mindspore.communication import (
    create_group,
    get_rank,
    get_local_rank,
    get_group_size,
    get_local_rank_size,
)


def run(world_size, rank):
    """Run net script for each process."""
    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(jit_level="O0")
    context.set_context(device_target="Ascend")

    print(f"===== pid: {os.getpid()}  rank: {rank} =====", flush=True)

    if rank in [0, 1, 2, 3]:
        ms.set_device(device_target="Ascend", device_id=rank)
        group_rank = rank % 4
        worker_num = 4
        init_process_group(
            init_method="tcp://127.0.0.1:12333", rank=group_rank, world_size=worker_num
        )

        assert get_rank() == rank
        assert get_local_rank() == rank
        assert get_group_size() == 4
        assert get_local_rank_size() == 4

    if rank in [4, 5]:
        ms.set_device(device_target="Ascend", device_id=rank)
        group_rank = rank % 4
        worker_num = 2
        init_process_group(
            init_method="tcp://127.0.0.1:12444", rank=group_rank, world_size=worker_num
        )

        assert get_rank() == rank % 4
        assert get_local_rank() == rank % 4
        assert get_group_size() == 2
        assert get_local_rank_size() == 2

    if rank in [6, 7]:
        ms.set_device(device_target="Ascend", device_id=rank)
        group_rank = rank % 6
        worker_num = 2
        init_process_group(
            init_method="tcp://127.0.0.1:12555", rank=group_rank, world_size=worker_num
        )

        assert get_rank() == rank % 6
        assert get_local_rank() == rank % 6
        assert get_group_size() == 2
        assert get_local_rank_size() == 2

    if rank in [0, 6, 7]:
        group_rank = rank % 5
        worker_num = 3
        init_process_group(
            init_method="tcp://127.0.0.1:12666",
            rank=group_rank,
            world_size=worker_num,
            group_name="init_for_067",
        )

    if rank == 0:
        assert get_rank() == 0
        assert get_local_rank() == 0
        assert get_group_size() == 4
        assert get_local_rank_size() == 4

    if rank == 6:
        assert get_rank() == rank % 6
        assert get_local_rank() == rank % 6
        assert get_group_size() == 2
        assert get_local_rank_size() == 2

    if rank in [0, 1, 2, 3]:
        create_group("self_group", [0, 1, 2, 3])

    if rank in [0, 1, 2, 3]:
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([12, 3]).astype(np.float32))
        output_handle = all_gather_into_tensor(output_tensor1, input_tensor1)
        except_output_tensor = cat(
            [input_tensor1, input_tensor1, input_tensor1, input_tensor1]
        )
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())

        input_tensor2 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor2 = ms.Tensor(np.zeros([12, 3]).astype(np.float32))
        output_handle = all_gather_into_tensor(
            output_tensor2, input_tensor2, group="self_group"
        )
        except_output_tensor = cat(
            [input_tensor2, input_tensor2, input_tensor2, input_tensor2]
        )
        assert output_handle is None
        assert np.allclose(output_tensor2.asnumpy(), except_output_tensor.asnumpy())

    if rank in [4, 5]:
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
        output_handle = all_gather_into_tensor(output_tensor1, input_tensor1)
        except_output_tensor = cat([input_tensor1, input_tensor1])
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())

    if rank in [6, 7]:
        input_tensor1 = ms.Tensor(np.arange(9).reshape(3, 3).astype(np.float32))
        output_tensor1 = ms.Tensor(np.zeros([6, 3]).astype(np.float32))
        output_handle = all_gather_into_tensor(output_tensor1, input_tensor1)
        except_output_tensor = cat([input_tensor1, input_tensor1])
        assert output_handle is None
        assert np.allclose(output_tensor1.asnumpy(), except_output_tensor.asnumpy())

    print(f"===== done pid: {os.getpid()}  rank: {rank} =====", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    rank_size = 8
    process_list = []
    for i in range(rank_size):
        p = mp.Process(target=run, args=(rank_size, i))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    for p in process_list:
        if p.exitcode != 0:
            raise RuntimeError(f"Process {p.pid} exits with exception! Error code: {p.exitcode}.")
