import os
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.ops import DataType, CustomRegOp
import mindspore.common.dtype as mstype
from mindspore.ops.composite import GradOperation
import pytest
from tests.mark_utils import arg_mark


class MultiScaleDeformableAttnFunc(nn.Cell):
    def __init__(self, out_shape=None, out_dtype=None):
        super().__init__()
        custom_msda_ref_info = CustomRegOp("aclnnMultiScaleDeformableAttn") \
            .input(0, "value", "required") \
            .input(1, "value_spatial_shapes", "required") \
            .input(2, "value_level_start_index", "required") \
            .input(3, "sampling_locations", "required") \
            .input(4, "attention_weights", "required") \
            .output(0, "output", "required") \
            .dtype_format(DataType.F32_Default,
                          DataType.I32_Default,
                          DataType.I32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()

        custom_msda_bprop_ref_info = CustomRegOp("aclnnMultiScaleDeformableAttnGrad") \
            .input(0, "value", "required") \
            .input(1, "spatial_shapes", "required") \
            .input(2, "level_start_index", "required") \
            .input(3, "sampling_loc", "required") \
            .input(4, "attn_weight", "required") \
            .input(5, "grad_output", "required") \
            .output(0, "grad_value", "required") \
            .output(1, "grad_sampling_loc", "required") \
            .output(2, "grad_attn_weight", "required") \
            .dtype_format(DataType.F32_Default,
                          DataType.I32_Default,
                          DataType.I32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default,
                          DataType.F32_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_msda_bprop = ops.Custom(
            "./infer_file/custom_cpp_infer.cc:aclnnMultiScaleDeformableAttnGrad",
            out_shape=out_shape,
            out_dtype=out_dtype,
            func_type="aot",
            bprop=None,
            reg_info=custom_msda_bprop_ref_info)

        def bprop(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                  out,
                  grad_output):
            grad_value, grad_sampling_loc, grad_attn_weight = self.custom_msda_bprop(value, value_spatial_shapes,
                                                                                     value_level_start_index,
                                                                                     sampling_locations,
                                                                                     attention_weights,
                                                                                     grad_output)
            return grad_value, None, None, grad_sampling_loc, grad_attn_weight

        self.custom_msda = ops.Custom("aclnnMultiScaleDeformableAttn",
                                      out_shape=lambda v_s, vss_s, vlsi_s, sl_s, aw_s: (
                                          v_s[0], sl_s[1], v_s[2] * v_s[3]),
                                      out_dtype=mstype.float32,
                                      func_type="aot",
                                      bprop=bprop,
                                      reg_info=custom_msda_ref_info)

    def construct(self, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
        return self.custom_msda(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                                attention_weights)


class TestMultiOutput():
    def setup_method(self):
        self.input_data_path = "/home/jiaorui/custom-multi-output/msdaData/msdaData"
        self.expect_data_path = "/home/jiaorui/custom-multi-output/baseline-result"
        self.value = Tensor(np.load(os.path.join(self.input_data_path, "value.npy"))).astype(ms.float32)
        self.shapes = Tensor(np.load(os.path.join(self.input_data_path, "spatial_shapes.npy"))).astype(ms.int32)
        self.level_start_index = Tensor(
            np.load(os.path.join(self.input_data_path, "level_start_index.npy"))).astype(ms.int32)
        self.sampling_locations = Tensor(
            np.load(os.path.join(self.input_data_path, "sampling_locations.npy"))).astype(ms.float32)
        self.attention_weights = Tensor(
            np.load(os.path.join(self.input_data_path, "attention_weights.npy"))).astype(ms.float32)

    def compare_out(self, grad_value, grad_sampling_loc, grad_attn_weight):
        base_grad_value = np.load(os.path.join(self.expect_data_path, "grad_value.npy"))
        assert np.allclose(grad_value.asnumpy(), base_grad_value)

        base_grad_samling_loc = np.load(os.path.join(self.expect_data_path, "grad_sampling_loc.npy"))
        assert np.allclose(grad_sampling_loc.asnumpy(), base_grad_samling_loc)

        base_grad_attn_weight = np.load(os.path.join(self.expect_data_path, "grad_attn_weight.npy"))
        assert np.allclose(grad_attn_weight.asnumpy(), base_grad_attn_weight)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
    @pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
    def test_cpp_infer_shape_type(self, context_mode):
        """
        Feature: Custom multi output testcase
        Description: cpp infer shape and cpp infer type
        Expectation: the result match with numpy result
        """
        context.set_context(mode=context_mode)
        msda_net = MultiScaleDeformableAttnFunc()
        grad = GradOperation(get_all=True, sens_param=False)(msda_net)
        grad_value, _, _, grad_sampling_loc, grad_attn_weight = grad(
            self.value, self.shapes, self.level_start_index, self.sampling_locations, self.attention_weights)
        self.compare_out(grad_value, grad_sampling_loc, grad_attn_weight)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
    @pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
    def test_python_infer_shape_type(self, context_mode):
        """
        Feature: Custom multi output testcase
        Description: python infer shape and python infer type
        Expectation: the result match with numpy result
        """
        context.set_context(mode=context_mode)
        msda_net = MultiScaleDeformableAttnFunc(out_shape=lambda v_s, vss_s, vlsi_s, sl_s, aw_s, go_s: [v_s,
                                                                                                        sl_s,
                                                                                                        [sl_s[0],
                                                                                                         sl_s[1],
                                                                                                         sl_s[2],
                                                                                                         sl_s[3],
                                                                                                         sl_s[4]]],
                                                out_dtype=[mstype.float32, mstype.float32, mstype.float32])
        grad = GradOperation(get_all=True, sens_param=False)(msda_net)
        grad_value, _, _, grad_sampling_loc, grad_attn_weight = grad(
            self.value, self.shapes, self.level_start_index, self.sampling_locations, self.attention_weights)
        self.compare_out(grad_value, grad_sampling_loc, grad_attn_weight)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
    @pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
    def test_cpp_infer_shape_python_infer_type(self, context_mode):
        """
        Feature: Custom multi output testcase
        Description: cpp infer shape and python infer type
        Expectation: the result match with numpy result
        """
        context.set_context(mode=context_mode)
        msda_net = MultiScaleDeformableAttnFunc(None, [mstype.float32, mstype.float32, mstype.float32])
        grad = GradOperation(get_all=True, sens_param=False)(msda_net)
        grad_value, _, _, grad_sampling_loc, grad_attn_weight = grad(
            self.value, self.shapes, self.level_start_index, self.sampling_locations, self.attention_weights)
        self.compare_out(grad_value, grad_sampling_loc, grad_attn_weight)

    @arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
    @pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
    def test_python_infer_shape_cpp_infer_type(self, context_mode):
        """
        Feature: Custom multi output testcase
        Description: python infer shape and cpp infer type
        Expectation: the result match with numpy result
        """
        context.set_context(mode=context_mode)
        msda_net = MultiScaleDeformableAttnFunc(out_shape=lambda v_s, vss_s, vlsi_s, sl_s, aw_s, go_s: [v_s,
                                                                                                        sl_s,
                                                                                                        [sl_s[0],
                                                                                                         sl_s[1],
                                                                                                         sl_s[2],
                                                                                                         sl_s[3],
                                                                                                         sl_s[4]]])
        grad = GradOperation(get_all=True, sens_param=False)(msda_net)
        grad_value, _, _, grad_sampling_loc, grad_attn_weight = grad(
            self.value, self.shapes, self.level_start_index, self.sampling_locations, self.attention_weights)
        self.compare_out(grad_value, grad_sampling_loc, grad_attn_weight)
