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
import tempfile

import numpy as np
import mindspore as ms
from mindspore._c_expression import MSContext

from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap


def get_max_relative_error(test_value, target_value):
    zero_mask = (target_value == 0)
    test_value[zero_mask] += np.finfo(float).eps
    target_value[zero_mask] += np.finfo(float).eps
    relative_err = np.divide((test_value - target_value), target_value)
    return np.max(np.abs(relative_err)).item()


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
    ms.set_context(mode=ms.PYNATIVE_MODE, jit_config={'jit_level': 'O0'}, max_device_memory="20GB")
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
    assert get_max_relative_error(output.numpy(), target_output.numpy()) <= 0.001
    assert get_max_relative_error(aux.numpy(), target_aux.numpy()) <= 0.001
    assert get_max_relative_error(grads[0].numpy(), target_grads[0].numpy()) <= 0.001
    assert get_max_relative_error(grads[1].numpy(), target_grads[1].numpy()) <= 0.001

    # check acldumpRegCallback in libmindspore_ascend.so.2
    script_dir = os.path.dirname(os.path.abspath(__file__))
    command = os.path.join(script_dir, 'check_adump_so.sh')
    assert os.system(f"bash {command}") == 0

    with tempfile.TemporaryDirectory(dir=script_dir) as tmp_dir:
        return_code = os.system(
            "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
            f"--master_port=10609 --join=True --log_dir={tmp_dir}/mstt_logs "
            f"python mstt_test_net.py {tmp_dir}"
        )
        assert return_code == 0

        target_values = {
            'forward_pre': np.asarray([[0.5, 0.5], [0.5, 0.5]]),
            'forward': np.asarray([[0.25, 0.25], [0.25, 0.25]]),
            'backward_pre': np.asarray([[4.0, 4.0], [4.0, 4.0]]),
            'backward': np.asarray([[2.0, 2.0], [2.0, 2.0]]),
            'tensor_grad': np.asarray([[2.0, 2.0], [2.0, 2.0]]),
            'param_grad': np.asarray([8.0]),
            'jit_forward': np.asarray([[0.5, 0.5], [0.5, 0.5]]),
            'jit_backward': np.asarray([[1.5, 1.5], [1.5, 1.5]])
        }
        ranks = ('0', '1', '2', '3')
        steps = ('0', '1')
        for rank in ranks:
            for step in steps:
                for data_type, target_value in target_values.items():
                    file_name = f'{tmp_dir}/rank_{rank}_step_{step}_{data_type}.npy'
                    if data_type == 'param_grad' and step == '1':
                        assert not os.path.isfile(file_name)
                    else:
                        assert os.path.isfile(file_name)
                        data_value = np.load(file_name)
                        assert get_max_relative_error(data_value, target_value) <= 0.001
