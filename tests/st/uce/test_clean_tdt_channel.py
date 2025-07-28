import mindspore as ms
from mindspore import Tensor, nn
from mindspore._c_expression import clean_tdt_channel
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_clean_tdt_channel():
    """
    Feature: clean tdt channel
    Description: Validate interface clean_tdt_channel
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
    x = Tensor(np.ones([1, 120, 1024, 640]), ms.float32)
    output = net(x)
    print(output.asnumpy().shape)

    ret = clean_tdt_channel()
    assert ret == 0
