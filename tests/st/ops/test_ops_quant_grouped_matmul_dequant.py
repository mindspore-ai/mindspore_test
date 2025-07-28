import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, ops
from tests.mark_utils import arg_mark

class QuantGroupedMatmulDequantNet(ms.nn.Cell):
    def __init__(self, quant_mode="pertoken"):
        super().__init__()
        self.quant_mode = quant_mode
        self.quant_gmm_dequant = ops.auto_generate.QuantGroupedMatmulDequant()

    def construct(self, x, weight, weight_scale, group_list, smooth_scale=None, x_scale=None):
        out = self.quant_gmm_dequant(x, weight, weight_scale, group_list, smooth_scale=smooth_scale,
                                     x_scale=x_scale, x_quant_mode=self.quant_mode)
        return out


def get_quant_grouped_matmul_dequant_golden(x, weight, weight_scale, group_list, smooth_scale=None, x_scale=None):
    G = weight.shape[0]
    x = x * smooth_scale if smooth_scale is not None else x
    if x_scale is None:
        x_scale = (ops.max(ops.abs(x.astype(ms.float32)), axis=-1)[0] / 127.0)
    else:
        x_scale = x_scale.repeat(x.shape[0])
    x_quantized = ops.round(x.astype(ms.float32) / x_scale.reshape(-1, 1)).astype(ms.int8)
    y_golden = []
    start_idx = 0
    for i in range(G):
        end_idx = group_list[i]
        xq = x_quantized[start_idx:end_idx].astype(ms.float32)
        wq = weight[i].astype(ms.float32)
        tmp = ops.matmul(xq, wq) * weight_scale[i] * x_scale[start_idx:end_idx].reshape(-1, 1)
        y_golden.append(tmp.astype(ms.float16))
        start_idx = end_idx
    y_golden = ops.concat(y_golden, axis=0)
    return y_golden


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_quant_grouped_matmul_dequant_pertonken(mode):
    """
    Feature: test quant_grouped_matmul_dequant operator
    Description: test matmul.
    Expectation: the result is correct
    """
    np.random.seed(12)
    ms.set_context(mode=mode)
    G = 4
    M = 64
    K = 256
    N = 512

    x = Tensor(np.random.randn(M, K).astype(np.float16))
    weight = Tensor(np.random.randint(-128, 127, size=(G, K, N)).astype(np.int8))
    weight_scale = Tensor(np.random.rand(G, N).astype(np.float32))
    group_list = Tensor(np.array([7, 29, 31, 64], dtype=np.int64))
    golden = get_quant_grouped_matmul_dequant_golden(x, weight, weight_scale, group_list)

    net = QuantGroupedMatmulDequantNet(quant_mode="pertoken")
    weight = weight.transpose(0, 2, 1).contiguous()
    out = net(x, weight, weight_scale, group_list)
    np.testing.assert_allclose(out.asnumpy(), golden.asnumpy(), rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend310p'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_quant_grouped_matmul_dequant_pertensor(mode):
    """
    Feature: test quant_grouped_matmul_dequant operator
    Description: test matmul.
    Expectation: the result is correct
    """
    np.random.seed(26)
    ms.set_context(mode=mode)
    G = 4
    M = 64
    K = 256
    N = 512

    x = Tensor(np.random.randn(M, K).astype(np.float16))
    weight = Tensor(np.random.randint(-128, 127, size=(G, K, N)).astype(np.int8))
    weight_scale = Tensor(np.random.rand(G, N).astype(np.float32))
    group_list = Tensor(np.array([7, 29, 31, 64], dtype=np.int64))
    x_scale = Tensor(np.random.randn(1).astype(np.float32))
    smooth_scale = Tensor(np.random.randn(K).astype(np.float16))
    golden = get_quant_grouped_matmul_dequant_golden(x, weight, weight_scale, group_list, smooth_scale, x_scale)

    net = QuantGroupedMatmulDequantNet(quant_mode="pertensor")
    weight = weight.transpose(0, 2, 1).contiguous()
    out = net(x, weight, weight_scale, group_list, smooth_scale, x_scale)
    np.testing.assert_allclose(out.asnumpy(), golden.asnumpy(), rtol=5e-3, atol=5e-3)
