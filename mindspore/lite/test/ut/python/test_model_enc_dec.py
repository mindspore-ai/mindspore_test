# Copyright 2023 Huawei Technologies Co., Ltd
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
Test lite python API.
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import One
import mindspore_lite as mslite

# ============================ ut testcases ============================
def test_mindir_export_load_with_encryption():
    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.dense_relu_sequential = nn.SequentialCell(
                nn.Dense(28*28, 512),
                nn.ReLU(),
                nn.Dense(512, 512),
                nn.ReLU(),
                nn.Dense(512, 10)
            )

        def construct(self, x):
            x = self.flatten(x)
            logits = self.dense_relu_sequential(x)
            return logits

    key = b'0123456789ABCDEF'
    model = Network()
    input_tensor = Tensor(shape=(1, 28, 28), dtype=ms.float32, init=One())
    output1 = np.array(model(input_tensor))
    ms.export(model, input_tensor, file_name="test_net", file_format="MINDIR", enc_key=key, enc_mode='AES-GCM')
    context = mslite.Context()
    context.target = ["cpu"]
    model_dec = mslite.Model()
    model_dec.build_from_file("./test_net.mindir", mslite.ModelType.MINDIR, context, dec_key=key, dec_mode='AES-GCM',
                              dec_num_parallel=2)
    inputs = model_dec.get_inputs()
    output2 = model_dec.predict(inputs)[0].get_data_to_numpy()
    assert np.all(output1 - output2) < 1e-6
