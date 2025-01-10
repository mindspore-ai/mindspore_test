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
import numpy as np
import pytest

import mindspore.context as context
from .optimizer_utils import FakeNet, build_network
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_default_asgd(mode):
    """
    Feature: Test SGD optimizer
    Description: Test SGD with group weight decay
    Expectation: Parameters conform to preset values.
    """
    from .optimizer_utils import default_fc1_weight_sgd, default_fc2_weight_sgd
    context.set_context(mode=mode)
    config = {'name': 'SGD', "weight_decay": 0.1}
    _, cells = build_network(config, FakeNet(), is_group=True)
    assert np.allclose(cells.accum[0].asnumpy(), default_fc1_weight_sgd, atol=1.e-4)
    assert np.allclose(cells.accum[2].asnumpy(), default_fc2_weight_sgd, atol=1.e-4)
