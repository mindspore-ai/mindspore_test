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
Test module for parallel training of Llama models using sharding propagation.
"""

import os
import sys
import argparse
import numpy as np
import mindspore as ms
from mindspore import set_seed
from mindspore.communication import init
from mindspore.dataset import GeneratorDataset

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers import Trainer, TrainingArguments

ms.set_context(jit_config={"jit_level": "O1"})
ms.set_context(mode=ms.GRAPH_MODE)
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL,
                             search_mode="sharding_propagation",
                             full_batch=True,
                             enable_parallel_optimizer=True)
init()


def generator_train():
    """train dataset generator"""
    seq_len = 4097
    step_num = 5
    batch_size = 32
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
                fine_grain_inteleave=1):
    """init task trainer."""
    set_seed(0)
    np.random.seed(0)

    args = TrainingArguments(
        batch_size=32, num_train_epochs=1, use_parallel=True)

    model_config = LlamaConfig(num_layers=80,
                               hidden_size=8192,
                               num_heads=64,
                               seq_length=4096,
                               batch_size=32,
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
                               fine_grain_inteleave=fine_grain_inteleave)
    model = LlamaForCausalLM(model_config)


    train_dataset = GeneratorDataset(
        generator_train, column_names=["input_ids"])
    train_dataset = train_dataset.batch(batch_size=32)

    task_trainer = Trainer(task='text_generation',
                           model=model,
                           args=args,
                           train_dataset=train_dataset)

    return task_trainer

def run_llama_compile():
    """test llama compile."""
    task_trainer = build_model('llama_compile', compute_dtype="float16")
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
    task_trainer.config.parallel.parallel_optimizer_config.optimizer_weight_shard_size = 1
    task_trainer.config.runner_config.gradient_accumulation_steps = 1
    ms.set_auto_parallel_context(pipeline_stages=8)
    task_trainer.set_parallel_config(data_parallel=1,
                                     model_parallel=4,
                                     context_parallel=1,
                                     pipeline_stage=8,
                                     micro_batch_num=32)
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
    if args.test_mode == "compile":
        run_llama_compile()


run_llama()
