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
Test module for parallel training of Llama models using Mindformers at jit_level O2.
"""

import os
import sys
import argparse
import numpy as np
import mindspore as ms
from mindspore import set_seed
from mindspore.dataset import GeneratorDataset
import mindspore.runtime as rt

workspace = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindformers.models.llama.llama_config import LlamaConfig
from mindformers.models.llama.llama import LlamaForCausalLM
from mindformers import Trainer, TrainingArguments


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
                param_init_type="float32",
                gradient_accumulation_steps=1,
                fine_grain_inteleave=1):
    """init task trainer."""
    set_seed(0)
    np.random.seed(0)

    args = TrainingArguments(
        batch_size=8, num_train_epochs=1, use_parallel=True)

    model_config = LlamaConfig(num_layers=2,
                               hidden_size=4096,
                               num_heads=32,
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
                               fine_grain_inteleave=fine_grain_inteleave)
    model = LlamaForCausalLM(model_config)


    train_dataset = GeneratorDataset(
        generator_train, column_names=["input_ids"])
    train_dataset = train_dataset.batch(batch_size=8)

    task_trainer = Trainer(task='text_generation',
                           model=model,
                           args=args,
                           train_dataset=train_dataset)

    return task_trainer

def run_llama_1p():
    """test llama no vmm"""
    task_trainer = build_model('llama_1p')
    task_trainer.config.callbacks[1].save_checkpoint_steps = 100
    task_trainer.config.callbacks = task_trainer.config.callbacks[:1]
    task_trainer.config.runner_config.epochs = 1
    task_trainer.config.runner_config.sink_mode = False
    task_trainer.config.runner_wrapper.scale_sense.loss_scale_value = 1024
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
    if args.test_mode == "no_vmm_kbk":
        ms.set_context(jit_config={"jit_level": "O0"})
        rt.set_memory(max_size="30GB")
        ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False)
        run_llama_1p()
    elif args.test_mode == "no_vmm_ge":
        ms.set_context(jit_config={"jit_level": "O2"})
        rt.set_memory(max_size="30GB")
        ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False)
        run_llama_1p()
    elif args.test_mode == "no_vmm_ge_two_pointer":
        ms.set_context(jit_config={"jit_level": "O2"})
        rt.set_memory(max_size="30GB")
        ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False)
        run_llama_1p()
    elif args.test_mode == "no_vmm_kbk_mempool_block":
        ms.set_context(jit_config={"jit_level": "O0"})
        rt.set_memory(max_size="30GB")
        ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False)
        run_llama_1p()
    elif args.test_mode == "no_vmm_kbk_oom":
        ms.set_context(jit_config={"jit_level": "O0"})
        rt.set_memory(max_size="30GB")
        ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False)
        run_llama_1p()

run_llama()
