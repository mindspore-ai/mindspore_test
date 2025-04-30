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

import os
import numpy as np
import shutil
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, Tensor, context, Parameter, train
from mindspore.communication import init
from mindspore.ops.auto_generate import WeightQuantBatchMatmul
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters

step_per_epoch = 4

train_sf_dir = "./parallel_weight"
strategy_file = "./strategy.ckpt"
unified_sf_dir = "./unified_weight"


class WeightQuantBatchMatmulNet(nn.Cell):
    """
    WeightQuantBatchMatmulNet.
    """

    def __init__(self, transpose_x=False, transpose_weight=False, antiquant_group_size=0, strategy=None):
        super().__init__()
        np_int4_weight = np.ones([8, 16]).astype(np.float16)
        self.wqbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight, antiquant_group_size)
        self.weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

        self.scale = 0.1
        self.offset = 4.0

        self.antiquant_scale = Tensor([self.scale], dtype=mstype.float16)
        self.antiquant_offset = Tensor([-self.offset], dtype=mstype.float16)
        self.quant_scale = None
        self.quant_offset = None
        self.bias = None

        if strategy is not None:
            self.wqbmm.shard(strategy)

    def construct(self, x):
        out = self.wqbmm(x, self.weight, self.antiquant_scale, self.antiquant_offset,
                         self.quant_scale, self.quant_offset, self.bias)
        return out


def get_dataset(*inputs):
    """Create dataset"""

    def generate():
        for _ in range(step_per_epoch):
            yield inputs

    return generate


def remove_ckpt_dir(file_path):
    """remove ckpt."""
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def train_model_with_2_rank(remove_redundancy):
    """
    Train parallel network with qint4x2
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, device_num=2,
                                      strategy_ckpt_config={"save_file": strategy_file})
    init()
    np.random.seed(1)
    input_data = np.random.rand(8, 8).astype(np.float16)
    label_data = np.random.rand(8, 32).astype(np.float16)

    fake_dataset = get_dataset(input_data, label_data)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # Set parallel strategy.
    strategy = None

    global_rank_id = int(os.getenv("RANK_ID"))
    ckpt_path = "./parallel_weight/rank_{}".format(global_rank_id)
    remove_ckpt_dir(ckpt_path)

    network = WeightQuantBatchMatmulNet(strategy=strategy)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    model = ms.Model(network=network, loss_fn=net_loss)
    ckpt_config = train.CheckpointConfig(keep_checkpoint_max=1, integrated_save=False, format="safetensors",
                                         remove_redundancy=remove_redundancy)
    ckpt_callback = train.ModelCheckpoint(prefix="parallel", directory=ckpt_path, config=ckpt_config)
    model.train(epoch=6, train_dataset=dataset, callbacks=[ckpt_callback], dataset_sink_mode=False)


def load_distributed_model(predict_data):
    """
    load_distributed_checkpoint with qint4x2
    """
    context.set_context(mode=context.GRAPH_MODE)
    ms.set_auto_parallel_context(full_batch=True, parallel_mode="semi_auto_parallel", device_num=2)
    init()
    print("distribute network shard.", flush=True)
    with no_init_parameters():
        network = WeightQuantBatchMatmulNet()
    print("distribute network create dataset.", flush=True)
    model = ms.Model(network)
    ms.load_distributed_checkpoint(network, predict_strategy=strategy_file,
                                   unified_safetensors_dir=unified_sf_dir, format="safetensors")
    predict_result = model.predict(ms.Tensor(predict_data))
    network2 = WeightQuantBatchMatmulNet()
    net_result = network2(ms.Tensor(predict_data))
    assert np.array_equal(predict_result.asnumpy(), net_result.asnumpy()), "result not equal, please check"


def test_train_model():
    '''
    Feature: Train parallel net.
    Description: Test ms.save_checkpoint
    Expectation: success.
    '''
    train_model_with_2_rank(remove_redundancy=False)


def test_train_model_remove_redundancy():
    '''
    Feature: Train parallel net.
    Description: Test ms.save_checkpoint with remove_redundancy.
    Expectation: success.
    '''
    train_model_with_2_rank(remove_redundancy=True)


def test_unified_sf():
    '''
    Feature: Merge safetensors.
    Description: Test ms.unified_safetensors.
    Expectation: success.
    '''
    remove_ckpt_dir(unified_sf_dir)
    ms.unified_safetensors(train_sf_dir, strategy_file, unified_sf_dir)


def test_unified_sf_remove_redundancy():
    '''
    Feature: Merge safetensors.
    Description: Test ms.unified_safetensors with remove_redundancy.
    Expectation: success.
    '''
    remove_ckpt_dir(unified_sf_dir)
    ms.unified_safetensors(train_sf_dir, strategy_file, unified_sf_dir,
                           merge_with_redundancy=False)


def test_load_distributed_predict_model():
    '''
    Feature: Parallel load and predict.
    Description: Test load_distributed_checkpoint.
    Expectation: success.
    '''
    predict_data = np.random.rand(8, 8).astype(np.float16)
    load_distributed_model(predict_data)
