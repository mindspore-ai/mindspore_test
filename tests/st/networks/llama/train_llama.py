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
"""
Test module for testing the paralleled llama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llama_model/test_parallel_train.py
pytest tests/st/test_model/test_llama_model/test_parallel_predict.py
"""

import os
import sys
import argparse
import numpy as np
import mindspore as ms
from mindspore import set_seed
from mindspore.communication import init
from mindspore.dataset import GeneratorDataset

workspace = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from training_checker import TrainingChecker
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers import Trainer, TrainingArguments

ms.set_context(jit_config={"jit_level": "O1"})
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                             gradients_mean=True,
                             full_batch=True,
                             enable_parallel_optimizer=True)
init()


def generator_train():
    """train dataset generator"""
    seq_len = 1025
    step_num = 10
    batch_size = 8
    vocab_size = 32000
    input_ids = np.random.randint(low=0, high=vocab_size, size=(
        step_num * batch_size, seq_len,)).astype(np.int32)
    for idx in range(len(input_ids)):
        yield input_ids[idx]


def build_model(test_mode,
                is_dynamic=False,
                compute_dtype="float16",
                softmax_compute_type="float32",
                layernorm_compute_type="float32",
                rotary_dtype="float32",
                param_init_type="float16",
                gradient_accumulation_steps=1,
                fine_grain_interleave=1):
    """init task trainer."""
    set_seed(0)
    np.random.seed(0)

    args = TrainingArguments(
        batch_size=8, num_train_epochs=1, use_parallel=True)

    model_config = LlamaConfig(num_layers=2,
                               hidden_size=1536,
                               num_heads=12,
                               seq_length=1024,
                               batch_size=8,
                               use_flash_attention=True,
                               use_past=False,
                               is_dynamic=is_dynamic,
                               compute_dtype=compute_dtype,
                               layernorm_compute_type=layernorm_compute_type,
                               softmax_compute_type=softmax_compute_type,
                               rotary_dtype=rotary_dtype,
                               param_init_type=param_init_type,
                               block_size=32,
                               num_blocks=20,
                               do_sample=False,
                               fine_grain_interleave=fine_grain_interleave)
    model = LlamaForCausalLM(model_config)


    train_dataset = GeneratorDataset(
        generator_train, column_names=["input_ids"])
    train_dataset = train_dataset.batch(batch_size=8)

    loss_list_std = [10.451367, 10.455378, 10.465119, 10.463621, 10.476261,
                     10.462841, 10.472476, 10.468395, 10.469678, 10.461041,]
    avg_step_time_std = 10000
    if test_mode == 'test_train_cp':
        loss_list_std = [10.448591, 10.450175, 10.458983, 10.466015, 10.473140,
                         10.459602, 10.472231, 10.466570, 10.462967, 10.467032,]
        avg_step_time_std = 10000
    if test_mode == 'test_train_dp':
        loss_list_std = [10.448593, 10.450171, 10.458986, 10.466034, 10.473145,
                         10.459610, 10.472258, 10.466605, 10.462999, 10.467015,]
        avg_step_time_std = 10000
    callback = TrainingChecker(loss_list_std=loss_list_std,
                               avg_step_time_std=avg_step_time_std,
                               micro_batch_num=2,
                               micro_batch_interleave_num=2,
                               gradient_accumulation_steps=gradient_accumulation_steps)

    task_trainer = Trainer(task='text_generation',
                           model=model,
                           args=args,
                           train_dataset=train_dataset,
                           callbacks=callback)

    return task_trainer


def run_llama_4p_train():
    """test msrun launch llama on 4p for Trainer.train()."""
    task_trainer = build_model('test_train', fine_grain_interleave=2)
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    task_trainer.set_parallel_config(data_parallel=1,
                                     model_parallel=2,
                                     pipeline_stage=2,
                                     micro_batch_num=2,
                                     micro_batch_interleave_num=2,
                                     vocab_emb_dp=False)
    task_trainer.train()
    sys.exit(0)


def run_llama_2p_train_cp():
    """test msrun launch llama on context parallel for Trainer.train()."""
    task_trainer = build_model('test_train_cp', gradient_accumulation_steps=2)
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    task_trainer.config.runner_config.gradient_accumulation_steps = 2
    task_trainer.set_parallel_config(data_parallel=1,
                                     model_parallel=1,
                                     context_parallel=2,
                                     pipeline_stage=1,
                                     micro_batch_num=2,
                                     micro_batch_interleave_num=2)
    task_trainer.train()
    sys.exit(0)


def run_llama_2p_train_dp():
    """test msrun launch llama on data parallel for Trainer.train()."""
    task_trainer = build_model('test_train_dp')
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    task_trainer.config.parallel.parallel_optimizer_config.optimizer_weight_shard_size = 1
    task_trainer.set_parallel_config(data_parallel=2,
                                     model_parallel=1,
                                     context_parallel=1,
                                     pipeline_stage=1,
                                     micro_batch_num=2,
                                     micro_batch_interleave_num=2)
    task_trainer.train()
    sys.exit(0)


def run_llama():
    """
    Feature: Trainer.train() Trainer.predict()
    Description: Test trainer for train/predict on parallel mode.
    Expectation: TypeError, ValueError, RuntimeError
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_mode', default="", type=str, help='test_mode.')
    args = parser.parse_args()
    if args.test_mode == "test_train":
        run_llama_4p_train()
    elif args.test_mode == "test_train_cp":
        run_llama_2p_train_cp()
    elif args.test_mode == "test_train_dp":
        run_llama_2p_train_dp()


run_llama()
