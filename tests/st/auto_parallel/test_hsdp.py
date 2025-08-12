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
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size
from mindspore import nn, ops
from mindspore.communication import init
from mindspore.parallel.spmd.hsdp import apply_hsdp

ms.set_context(mode=ms.PYNATIVE_MODE)
init()
ms.set_seed(1)

def create_dataset(batch_size):
    dataset_path = "/home/workspace/mindspore_dataset/mnist/train"
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.MnistDataset(dataset_path, num_shards=rank_size, shard_id=rank_id)
    image_transforms = [
        ds.vision.Rescale(1.0 / 255.0, 0),
        ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ds.vision.HWC2CHW()
    ]
    label_transform = ds.transforms.TypeCast(ms.int32)
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

local_batch_size = 32
data_set = create_dataset(local_batch_size)

class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

loss_fn = nn.CrossEntropyLoss()

def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

def hsdp_without_accumulate_grad(shard_size, threshold=64, optimizer_level="level1"):
    net = Network()
    apply_hsdp(net, shard_size, threshold, optimizer_level)

    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)

    i = 0
    rank_id = get_rank()
    for data, label in data_set:
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        if rank_id == 0 and i % 10 == 0:
            print("step: %s, loss is %s" % (i, loss))
        i += 1

def hsdp_with_accumulate_grad(shard_size, threshold=64, optimizer_level="level1", micro_step=1):
    net = Network()
    apply_hsdp(net, shard_size, threshold, optimizer_level, accumulate_grad_step=micro_step)

    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)

    i = 0
    rank_id = get_rank()
    micro_size = local_batch_size // micro_step
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
        optimizer(grads)
        if rank_id == 0 and i % 10 == 0:
            print("step: %s, loss is %s" % (i, total_loss / micro_step))
        i += 1

def test_pure_dp():
    hsdp_without_accumulate_grad(shard_size=1)

def test_zero1_fully_shard():
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level1")

def test_zero1_partial_shard():
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level1")

def test_zero2_fully_shard():
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level2")

def test_zero2_partial_shard():
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level2")

def test_zero3_fully_shard():
    hsdp_without_accumulate_grad(shard_size=8, optimizer_level="level3")

def test_zero3_partial_shard():
    hsdp_without_accumulate_grad(shard_size=4, optimizer_level="level3")

def test_pure_dp_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=1, micro_step=8)

def test_zero1_fully_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level1", micro_step=8)

def test_zero1_partial_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level1", micro_step=8)

def test_zero2_fully_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level2", micro_step=8)

def test_zero2_partial_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level2", micro_step=8)

def test_zero3_fully_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=8, optimizer_level="level3", micro_step=8)

def test_zero3_partial_shard_with_acc_grad():
    hsdp_with_accumulate_grad(shard_size=4, optimizer_level="level3", micro_step=8)
