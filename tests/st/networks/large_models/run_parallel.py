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

import numpy as np

from similarity import compare_distance

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "networks/mindformers"))

import mindspore as ms
from mindspore import Tensor, Model, mint, set_seed
from mindspore.common import initializer as init
from mindspore.nn.utils import no_init_parameters

from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig, LlamaForCausalLM
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.pet import get_pet_model
from mindformers.pet.pet_config import SLoraConfig
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
            compare_distance(output_text, answer, bench_sim=0.95)


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
            compare_distance(output_text, answer, bench_sim=0.95)


def parallel_qwen2_0_5b_parallel_decoding_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "PagedAttention,FlashAttentionScore"
    os.environ["RUN_MODE"] = "predict"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

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
    config.model.model_config.use_past = True
    config.model.model_config.use_flash_attention = True
    config.model.model_config.is_dynamic = True
    config.model.model_config.parallel_decoding_params = {"parallel_decoding": "la"}

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

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
    network.set_dynamic_inputs()

    # forward
    inputs_list = []
    for i in range(5):
        bs = i + 1
        seq_len = i + 2
        token_num = bs * seq_len
        batch_valid_length = seq_len if i == 0 else i + 3

        input_ids = np.arange(0, token_num, dtype=np.int32)
        input_ids[input_ids % (i + 1)] = 1
        position_ids = np.arange(0, token_num, dtype=np.int32)
        position_ids[position_ids % (i + 1)] = 0
        spec_mask = np.arange(0, token_num * batch_valid_length, dtype=np.float16)
        spec_mask[spec_mask % (i + 1) != 0] = 0

        inputs = {
            "input_ids": input_ids,
            "valid_length_each_example": np.array([batch_valid_length] * bs, np.int32),
            "block_tables": np.arange(0, stop=int(model_config.num_blocks // bs * bs), dtype=np.int32).reshape(bs, -1),
            "slot_mapping": np.arange(0, stop=token_num, dtype=np.int32),
            "position_ids": position_ids,
            "spec_mask": spec_mask.reshape(token_num, batch_valid_length),
            "q_seq_lens": np.array([seq_len] * bs, np.int32),
            "use_past": True,
            "prefill": i == 0,
        }
        inputs_list.append(inputs)

    expect_list = [
        np.array([
            [24184, 18493],
        ], np.int32),
        np.array([
            [1, 91997],
            [1, 91997],
            [1, 698],
            [121487, 235],
            [95655, 95113],
            [110616, 9179],
        ], np.int32),
        np.array([
            [1, 698],
            [1, 3],
            [1, 698],
            [1, 698],
            [95655, 95113],
            [110616, 9179],
            [34509, 46423],
            [69183, 75116],
            [3591, 116053],
            [112738, 13845],
            [13, 220],
            [56852, 87973],
        ], np.int32),
        np.array([
            [3014, 1],
            [3014, 220],
            [38646, 10892],
            [3014, 220],
            [67, 35],
            [110616, 9179],
            [34509, 46423],
            [69183, 75116],
            [3591, 116053],
            [112738, 19310],
            [13, 220],
            [56852, 87973],
            [75116, 75694],
            [103351, 101724],
            [100239, 26797],
            [50238, 131275],
            [62785, 26916],
            [104512, 2742],
            [24695, 101911],
            [19264, 32638],
        ], np.int32),
        np.array([
            [16, 17],
            [12, 220],
            [2130, 16],
            [12, 220],
            [220, 15],
            [68590, 33624],
            [34509, 46423],
            [69183, 75116],
            [3591, 116053],
            [112738, 19310],
            [13, 220],
            [56852, 87973],
            [75116, 75694],
            [103351, 101724],
            [100239, 26797],
            [50238, 131275],
            [62785, 26916],
            [104512, 2742],
            [24695, 101911],
            [19264, 32638],
            [10, 13],
            [102608, 115698],
            [90867, 10402],
            [8937, 102347],
            [60919, 9442],
            [30858, 93304],
            [13, 220],
            [75116, 94892],
            [106004, 114898],
            [17, 94443],
        ], np.int32),
    ]
    for inputs, expect in zip(inputs_list, expect_list):
        res, _ = network.forward(**inputs)
        _, indices = mint.topk(res, 2)
        print(f'res {res.shape}\n', res)
        print(f'indices {indices.shape}\n', indices)
        if inputs["prefill"]:
            expect_shape = (inputs["valid_length_each_example"].shape[0], model_config.vocab_size)
        else:
            expect_shape = (inputs["input_ids"].shape[0], model_config.vocab_size)
        print(f'expect_shape {expect_shape}')
        assert res.shape == expect_shape
        assert np.allclose(indices.numpy().astype(np.int32), expect, atol=1e-3)


def parallel_qwen2_0_5b_multilora_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "qwen/configs/ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    set_seed(0)
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
    model_config.pet_config = SLoraConfig(lora_num=2, lora_rank=8, lora_alpha=16,
                                          target_modules='.*wq|.*wk|.*wv|.*wo|.*w1|.*w2|.*w3')

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    with no_init_parameters():
        network = LlamaForCausalLM(model_config)
        network = get_pet_model(network, model_config.pet_config)
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
            print("parallel_qwen2_0_5b_multilora_mp2, output_text:", output_text)
            compare_distance(output_text, answer, bench_sim=0.95)


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_qwen2_0_5b_predict_mp2_static': parallel_qwen2_0_5b_predict_mp2_static,
    'parallel_qwen2_0_5b_parallel_decoding_mp2': parallel_qwen2_0_5b_parallel_decoding_mp2,
    'parallel_qwen2_0_5b_multilora_mp2': parallel_qwen2_0_5b_multilora_mp2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
