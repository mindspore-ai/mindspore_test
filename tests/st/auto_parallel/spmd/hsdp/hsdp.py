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
from typing import Optional
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size
from mindspore import nn, ops
from mindspore.communication import init
from mindspore.parallel import hsdp
from hsdp_test_common import hsdp_network_ckpt_path
from tests.mark_utils import arg_mark
from tests.st.auto_parallel.spmd.common_net import SlimLeNet

ms.set_seed(1)
ms.set_deterministic(True)

def create_dataset(local_batch_size: int, num_shards: Optional[int] = None, shard_id: Optional[int] = None):
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
    if (num_shards is None) or (shard_id is None):
        dataset = ds.MnistDataset(dataset_path, shuffle=False)
    else:
        dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id, shuffle=False)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(local_batch_size)
    return dataset


loss_fn = nn.CrossEntropyLoss()
def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

# Global hyper parameters:
local_bs = 32
dp_size = get_group_size()
rank_id = get_rank()
rank_size = get_group_size()
learning_rate = 1e-3
max_step = 10

def make_baseline_by_standalone_run():
    data_set = create_dataset(local_batch_size=local_bs * dp_size)
    net = SlimLeNet()
    param_dict = ms.load_checkpoint(hsdp_network_ckpt_path)
    param_not_load, _ = ms.load_param_into_net(net, param_dict)
    assert not param_not_load, f"For hsdp test case, not completely load ckpt from {hsdp_network_ckpt_path}"
    optimizer = nn.Adam(net.trainable_params(), learning_rate)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)

    i = 0
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        if rank_id == 0:
            print(f"step: {i}, loss: {loss}")
        i += 1
        if i >= max_step:
            break

def hsdp_without_accumulate_grad(shard_size, threshold=64, optimizer_level="level1"):
    data_set = create_dataset(local_batch_size=local_bs, num_shards=dp_size, shard_id=rank_id)
    net = SlimLeNet()
    hsdp(net, shard_size, threshold, optimizer_level)
    optimizer = nn.Adam(net.trainable_params(), learning_rate)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)
    i = 0
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        reduced_loss = loss_sync_allreduce(loss)
        if rank_id == 0:
            print(f"step: {i}, loss: {reduced_loss / dp_size}")
        i += 1
        if i >= max_step:
            break

def hsdp_with_accumulate_grad(shard_size, threshold=64, optimizer_level="level1", micro_step=1):
    data_set = create_dataset(local_batch_size=local_bs, num_shards=dp_size, shard_id=rank_id)
    net = SlimLeNet()
    hsdp(net, shard_size, threshold, optimizer_level, enable_grad_accumulation=True)
    optimizer = nn.Adam(net.trainable_params(), learning_rate)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    loss_sync_allreduce = ops.AllReduce(ops.ReduceOp.SUM)
    i = 0
    micro_size = local_bs // micro_step
    for data, label in data_set:
        data_list = ops.split(data, micro_size)
        label_list = ops.split(label, micro_size)
        if len(data_list) < micro_step:
            continue
        net.zero_grads()
        net.set_requires_grad_sync(False)
        total_loss = 0
        for j in range(micro_step):
            if j == micro_step - 1:
                net.set_requires_grad_sync(True)
            (loss, _), grads = grad_fn(data_list[j], label_list[j])
            total_loss = total_loss + loss
        reduced_loss = loss_sync_allreduce(total_loss)
        optimizer(grads)
        if rank_id == 0:
            print(f"step: {i}, loss: {reduced_loss / (micro_step * dp_size)}")
        i += 1
        if i >= max_step:
            break

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_standalone_run():
    '''
    Feature: Run the network with standalone mode
    Description: run network with standalone, gbs = local_batch_size * dp_size
    Expectation: Run success
    '''
    make_baseline_by_standalone_run()

def test_pure_dp():
    init()
    hsdp_without_accumulate_grad(shard_size=1)

def test_zero1_fully_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level1")

def test_zero1_partial_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level1")

def test_zero2_fully_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level2")

def test_zero2_partial_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level2")

def test_zero3_fully_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level3")

def test_zero3_partial_shard():
    init()
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level3")

def test_pure_dp_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=1, micro_step=8)

def test_zero1_fully_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level1", micro_step=8)

def test_zero1_partial_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level1", micro_step=8)

def test_zero2_fully_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level2", micro_step=8)

def test_zero2_partial_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level2", micro_step=8)

def test_zero3_fully_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level3", micro_step=8)

def test_zero3_partial_shard_with_acc_grad():
    init()
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level3", micro_step=8)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_no_dp():
    '''
    Feature: apply hsdp with single node.
    Description: apply hsdp with single node.
    Expectation: Run success
    '''
    hsdp_without_accumulate_grad(shard_size=1)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_no_dp_with_acc_grad():
    '''
    Feature: apply hsdp with single node gradient accumulation.
    Description: apply hsdp with single node gradient accumulation.
    Expectation: Run success
    '''
    hsdp_with_accumulate_grad(shard_size=1, micro_step=8)
