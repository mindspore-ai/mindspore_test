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
import argparse
import sys
import os
import numpy as np
import time

workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(workspace, "mindformers"))
from mindformers import LlamaConfig, TransformerOpParallelConfig, LlamaForCausalLM, build_context
from mindformers.tools.register import MindFormerConfig
from mindspore import set_seed
from mindspore.nn.utils import no_init_parameters
from infer_qwen import generate_input_ids


def build_model(config_path, batch_size=1, model_parallel=1, use_bf16=False):
    set_seed(0)
    np.random.seed(0)
    # set model config
    config = MindFormerConfig(config_path)

    if model_parallel == 1:
        config.use_parallel = False
        device_id = int(os.getenv('DEVICE_ID', '0'))
        config.context.device_id = device_id
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
    return model_config


def run_qwen_1p_bs1(args):
    model_config = build_model(args.yaml_file, batch_size=1, model_parallel=1)
    # build model from config
    build_model_start = time.time()
    with no_init_parameters():
        model = LlamaForCausalLM(model_config)
    build_model_end = time.time()
    assert (build_model_end - build_model_start) < 80
    model.set_train(False)

    inputs_ids = generate_input_ids(1, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 112013, 58939, 26026,
                           120500, 90532, 65153, 50947, 91544, 121978, 54324], dtype=np.int32)
    assert (EXPECT_RES == outputs).all()


def run_qwen_4p_bs1(args):
    model_config = build_model(args.yaml_file, batch_size=1, model_parallel=4)
    build_model_start = time.time()
    with no_init_parameters():
        model = LlamaForCausalLM(model_config)
    build_model_end = time.time()
    assert (build_model_end - build_model_start) < 15
    model.set_train(False)

    inputs_ids = generate_input_ids(4, 10)
    outputs = model.generate(inputs_ids, max_length=20, do_sample=False)
    EXPECT_RES = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    for output in outputs:
        assert (EXPECT_RES == output).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default="", type=str,
                        help='predict yaml path')
    parser.add_argument('--test_mode', default='', type=str,
                        help='test mode.')
    args_ = parser.parse_args()
    test_mode = args_.test_mode
    if test_mode == "test_qwen_1p_bs1":
        run_qwen_1p_bs1(args_)
    if test_mode == "test_qwen_4p_bs1":
        run_qwen_4p_bs1(args_)
