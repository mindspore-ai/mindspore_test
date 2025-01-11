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
Test module for testing the paralleled infer interface used for mindformers.
How to run this:
    pytest tests/st/networks/large_models/test_parallel_predict.py
"""
import argparse
import os
import sys

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "networks/mindformers"))

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init
from mindspore.nn.utils import no_init_parameters

from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig, LlamaForCausalLM
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer


def parallel_qwen2_0_5b_predict_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./qwen2_05b_dynamic_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n以下是一个使用Python实现的快速排序"
                                 "算法：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        "
                                 "return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] "
                                 "if x < pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        "
                                 "return quick_sort(left) + [pivot] + quick_sort(right)\n\n# 示例输入\narr = [3,6,8,1"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of "
                                 "life is a philosophical question that has been debated for centuries, and there "
                                 "is no one definitive answer to it. Some people believe that the meaning of life "
                                 "is to find happiness and fulfillment in their lives, while others believe that it "
                                 "is to achieve success or recognition.\n\nOthers may argue that the meaning of life "
                                 "is to make a positive impact on the world, to help others, and to contribute to "
                                 "society as a whole. Others may believe that the meaning of life is to pursue "
                                 "knowledge and understanding, to"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_mp2, output_text:", output_text)
            assert output_text == answer


def parallel_qwen2_0_5b_predict_mp2_static():
    """test qwen2-0.5B predict in model_parallel=2 with static shape"""
    # config.model.model_config.is_dynamic = False
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_config.save_file = "./qwen2_05b_static_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.is_dynamic = False
    config.model.model_config.do_sample = False
    config.model.model_config.temperature = 1.0
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {4: {"prompt": "你好!",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "你好!<|im_end|>\n<|im_start|>assistant\n你好！很高兴为你提供帮助。"
                                 "有什么我可以帮助你的吗？<|im_end|>"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)
        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_mp2_static, output_text:", output_text)
            assert output_text == answer


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_qwen2_0_5b_predict_mp2_static': parallel_qwen2_0_5b_predict_mp2_static,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
