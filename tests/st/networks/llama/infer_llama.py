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
import sys
import os
import argparse
import pandas as pd
import numpy as np
from mindspore import set_seed

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
sys.path.insert(0, os.path.join(workspace, "golden-stick"))
from mindformers.tools.register import MindFormerConfig
from mindformers import LlamaConfig, TransformerOpParallelConfig, LlamaForCausalLM, build_context

TOELERANCE = 5e-2


def str2bool(b):
    """String convert to Bool."""
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def generate_input_ids(batch_size, input_seq_lenth):
    return [[i+1 for i in range(input_seq_lenth)] for _ in range(batch_size)]


def get_total_time_from_profiler_file(file_path):
    if not file_path:
        raise FileExistsError(f"profiler file not exist")
    op_static = pd.read_csv(file_path)
    total_time = op_static['Total Time(us)'].sum()
    return total_time


def build_model(config_path, batch_size=1, model_parallel=1, use_bf16=False, quant=None):
    set_seed(100)
    np.random.seed(100)
    # set model config
    config = MindFormerConfig(config_path)
    if quant:
        config.quant = quant

    if model_parallel == 1:
        config.use_parallel = False
    else:
        # set parallel method
        config.use_parallel = True
        config.parallel_config.data_parallel = 1
        config.parallel_config.model_parallel = model_parallel
        config.parallel_config.pipeline_stage = 1
        print(config.parallel_config)

    if use_bf16:
        config.model.model_config.compute_dtype = "bfloat16"
        config.model.model_config.layernorm_compute_type = "bfloat16"
        config.model.model_config.softmax_compute_type = "bfloat16"
        config.model.model_config.rotary_dtype = "bfloat16"
        config.model.model_config.param_init_type = "bfloat16"

    # initialize env
    build_context(config)
    model_config = LlamaConfig(**config.model.model_config)

    # set model parameters
    model_config.parallel_config = TransformerOpParallelConfig(
        **config.parallel_config)
    model_config.batch_size = batch_size

    # build model from config
    model = LlamaForCausalLM(model_config)
    model.set_train(False)
    return model


def run_llama_1p_bs1(args):
    model = build_model(args.yaml_file, batch_size=1, model_parallel=1)

    inputs_ids = generate_input_ids(1, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19434, 14518,
                           11470, 6527, 13385, 9334, 9551, 29535, 19805, 3270], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == outputs).all()

    inputs_ids = generate_input_ids(4, 12)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17985,
                           25267, 6935, 30170, 30901, 28619, 30901, 30901], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_llama_1p_bs4(args):
    model = build_model(args.yaml_file, batch_size=4, model_parallel=1)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19434, 14518,
                           11470, 6527, 13385, 9334, 9551, 29535, 19805, 3270], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17985,
                           25267, 6935, 30170, 30901, 28619, 30901, 30901], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_llama_4p_bs1(args):
    model = build_model(args.yaml_file, batch_size=1, model_parallel=4)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8631, 21301,
                           9393, 6950, 5321, 26787, 8752, 18897, 21524, 22538], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16461,
                           31007, 24750, 4468, 4775, 25799, 18814, 11507], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_llama_4p_bs4(args):
    model = build_model(args.yaml_file, batch_size=4, model_parallel=4)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8631, 21301,
                           9393, 6950, 5321, 26787, 8752, 18897, 21524, 22538], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()

    inputs_ids = generate_input_ids(8, 12)
    outputs = model.generate(inputs_ids,
                             max_length=20,
                             do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16461,
                           31007, 24750, 4468, 4775, 25799, 18814, 11507], dtype=np.int32)

    for output in outputs:
        assert (EXPECT_RES == output).all()


def run_llama_4p_bs4_bf16(args):
    model = build_model(args.yaml_file, batch_size=4,
                        model_parallel=4, use_bf16=True)

    inputs_ids = generate_input_ids(4, 10)
    model.generate(inputs_ids, max_length=20, do_sample=False)

    inputs_ids = generate_input_ids(8, 12)
    model.generate(inputs_ids, max_length=20, do_sample=False)


def run_llama_4p_bs4_w8a16(args):
    model = build_model(args.yaml_file, batch_size=4,
                        model_parallel=4, use_bf16=True, quant="w8a16")

    inputs_ids = generate_input_ids(4, 10)
    model.generate(inputs_ids, max_length=20, do_sample=False)

    inputs_ids = generate_input_ids(8, 12)
    model.generate(inputs_ids, max_length=20, do_sample=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--test_mode', default='', type=str,
                        help='test mode.')
    args_ = parser.parse_args()
    test_mode = args_.test_mode
    if test_mode == "test_llama_1p_bs1":
        run_llama_1p_bs1(args_)
    if test_mode == "test_llama_1p_bs4":
        run_llama_1p_bs4(args_)
    if test_mode == "test_llama_4p_bs1":
        run_llama_4p_bs1(args_)
    if test_mode == "test_llama_4p_bs4":
        run_llama_4p_bs4(args_)
    if test_mode == "test_llama_4p_bs4_bf16":
        run_llama_4p_bs4_bf16(args_)
    if test_mode == "test_llama_4p_bs4_w8a16":
        run_llama_4p_bs4_w8a16(args_)
