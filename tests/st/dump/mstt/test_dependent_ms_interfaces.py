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
import shutil
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap

import mindspore as ms
from mindspore._c_expression import MSContext


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
@security_off_wrap
def test_interfaces_used_in_mstt():
    """
    Feature: MindSpore interfaces used in msprobe.
    Description: Test MindSpore interfaces used in msprobe.
    Expectation: no error.
    """

    soc_version = MSContext.get_instance().get_ascend_soc_version()
    assert 'ascend' in soc_version

    ms.set_device('Ascend')
    ms.set_context(mode=ms.PYNATIVE_MODE, jit_config={'jit_level': 'O0'})
    mode = ms.get_context("mode")
    jit_level = ms.context.get_jit_config().get("jit_level")
    assert mode == ms.PYNATIVE_MODE
    assert jit_level == 'O0'

    def fn(x, y):
        return 2 * x + y, y ** 3
    x = ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    y = ms.Tensor([[1, 2], [3, 4]], dtype=ms.float32)
    v = ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32)
    target_output = ms.Tensor([[3, 6], [9, 12]], dtype=ms.float32)
    target_aux = ms.Tensor([[1, 8], [27, 64]], dtype=ms.float32)
    target_grads = (ms.Tensor([[2, 2], [2, 2]], dtype=ms.float32),
                    ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32))
    output, vjp_fn, aux = ms.vjp(fn, x, y, has_aux=True)
    grads = vjp_fn(v)
    assert (output - target_output).abs().sum().item() < 0.001
    assert (aux - target_aux).abs().sum().item() < 0.001
    assert (grads[0] - target_grads[0]).abs().sum().item() < 0.001
    assert (grads[1] - target_grads[1]).abs().sum().item() < 0.001

    if os.path.isdir('./data'):
        shutil.rmtree('./data')
    os.mkdir('./data')

    return_code = os.system(
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
        "--master_port=10609 --join=True --log_dir=./mstt_logs "
        "python mstt_test_net.py"
    )
    assert return_code == 0

    data_types = ('forward_pre', 'forward', 'backward_pre', 'backward', 'tensor_grad', 'param_grad',
                  'jit_forward', 'jit_backward')
    ranks = ('0', '1', '2', '3')
    steps = ('0', '1')
    for rank in ranks:
        for step in steps:
            for data_type in data_types:
                file_name = f'./data/rank_{rank}_step_{step}_{data_type}.npy'
                if data_type == 'param_grad' and step == '1':
                    assert not os.path.isfile(file_name)
                else:
                    assert os.path.isfile(file_name)

    shutil.rmtree('./data')
