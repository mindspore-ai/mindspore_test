/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "infer/renorm.h"
#include "infer/scatter_update.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace expander {
namespace {
bool IsLastAxis(const ShapeVector &shape, int64_t axis) {
  if (axis == -1) {
    return true;
  }
  if (IsDynamicRank(shape)) {
    return false;
  }
  auto rank = SizeToLong(shape.size());
  if (axis < 0) {
    axis += rank;
  }
  return (axis == (rank - 1));
}

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  std::vector<int64_t> reverse_axis;
  if (x_shape.empty()) {
    return reverse_axis;
  }
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  reverse_axis.reserve(x_shape.size());
  for (int64_t i = 0; i < rk; ++i) {
    (void)reverse_axis.emplace_back(i);
  }
  reverse_axis[LongToSize(axis)] = rk - 1;
  reverse_axis[LongToSize(rk - 1)] = axis;
  return reverse_axis;
}
}  // namespace

REG_FALLBACK_BUILDER("ArgMaxExt").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto keepdim = ib->GetInput(kIndex2);
  if (input_x->dtype()->type_id() == kBool->type_id()) {
    input_x = ib->Cast(input_x, kInt32);
  }
  bool is_dim_none = True;
  auto dim_value = ib->Value<int64_t>(0);
  auto dim_value_ptr = dim->BuildValue();
  if (dim_value_ptr->isa<None>()) {
    input_x = ib->Reshape(input_x, {-1});
  } else {
    dim_value = dim;
    is_dim_none = False;
  }
  auto res = ib->Emit("Argmax", {input_x, dim_value, ib->Value<int64_t>(kInt64->type_id())});
  auto keepdim_value = GetScalarValue<bool>(keepdim->BuildValue());
  if (!keepdim_value.has_value()) {
    auto true_case = [&res, &dim](Emitter *e) -> NodePtrList { return {e->Emit("ExpandDims", {res, dim})}; };
    auto false_case = [&res](Emitter *e) -> NodePtrList { return {res}; };
    if (!is_dim_none) {
      res = ib->Conditional(keepdim, true_case, false_case);
    }
  } else {
    if (keepdim_value.value() && !is_dim_none) {
      res = ib->Emit("ExpandDims", {res, dim});
    }
  }
  return {res};
});

REG_FALLBACK_BUILDER("SiLU").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto s = ib->Emit("Sigmoid", {ib->GetInput(kIndex0)});
  return {ib->Mul(input_x, s)};
});

REG_FALLBACK_BUILDER("SiLUGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto sigmoid_input = ib->Emit("Sigmoid", {x});
  auto bc_dx = ib->Mul(x, dout);
  auto bc_dy = ib->Mul(sigmoid_input, dout);
  auto dx = ib->Emit("SigmoidGrad", {sigmoid_input, bc_dx});
  return {ib->Add(dx, bc_dy)};
});

REG_FALLBACK_BUILDER("Baddbmm").SetBody(BODYFUNC(ib) {
  // baddbmm equation: output = beta * input + alpha * matmul(batch1, batch2)
  auto input = ib->GetInput(kIndex0);
  auto batch1 = ib->GetInput(kIndex1);
  auto batch2 = ib->GetInput(kIndex2);
  auto beta = ib->GetInput(kIndex3);
  auto alpha = ib->GetInput(kIndex4);

  auto mm_output = ib->BatchMatMul(batch1, batch2);
  auto alpha_output = ib->Mul(mm_output, alpha);
  auto beta_output = ib->Mul(input, beta);
  return {ib->Add(beta_output, alpha_output)};
});

DEF_PURE_SHAPE_CALC(g_dense_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(kIndex0);
    auto &w_shape = inputs.at(kIndex1);
    ShapeVector reshape_x_shape = {-1, x_shape.back()};
    ShapeVector reshape_w_shape = {-1, w_shape.back()};
    ShapeVector reshape_ret_shape;
    reshape_ret_shape = x_shape;
    if (w_shape.size() == 1) {
      reshape_ret_shape.erase(reshape_ret_shape.end() - 1);
    } else {
      reshape_ret_shape.back() = w_shape[kIndex0];
    }
    return {reshape_x_shape, reshape_w_shape, reshape_ret_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    constexpr const int64_t kRank2 = 2;
    int64_t ret_size = -1LL;
    if (!IsDynamicRank(inputs[0]) && !IsDynamicRank(inputs[1])) {
      if (inputs[0].size() == 1) {
        if (inputs[1].size() == 1) {
          ret_size = 0;
        } else {
          ret_size = 1;
        }
      } else {
        ret_size = inputs[0].size();
      }
    }
    return {kRank2, kRank2, ret_size};
  });

REG_FALLBACK_BUILDER("Dense").SetBody(BODYFUNC(ib) {
  constexpr const size_t kRank2 = 2;
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  NodePtrList reshape_shapes;
  auto x_shape = x->shape();
  auto w_shape = w->shape();
  bool is_empty_tensor = x_shape.size() == 1 && w_shape.size() == 1 && x_shape[0] == 0 && w_shape[0] == 0;
  if (is_empty_tensor) {
    return {ib->GetInput(kIndex2)};
  }
  bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
  bool need_reshape = (is_dynamic_rank || x_shape.size() != kRank2 || w_shape.size() != kRank2);
  if (need_reshape) {
    reshape_shapes = ib->ShapeCalc(g_dense_shapecalc, {x, w});
    x = ib->Reshape(x, reshape_shapes[kIndex0]);
    w = ib->Reshape(w, reshape_shapes[kIndex1]);
  }
  auto ret = ib->MatMul(x, w, false, true);
  ret = ib->Cast(ret, x->dtype());
  auto b = ib->GetInput(kIndex2);
  auto b_value = b->BuildValue();
  if (!b_value->isa<None>()) {
    ret = ib->Add(ret, b);
  }
  if (need_reshape) {
    ret = ib->Reshape(ret, reshape_shapes[kIndex2]);
  }
  return {ret};
});

DEF_PURE_SHAPE_CALC(g_nonzero_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(kIndex0);
    ShapeVector reshape_x_shape = x_shape;
    if (x_shape.size() == 0) {
      reshape_x_shape.push_back(kIndex1);
    }
    return {reshape_x_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    int64_t ret_size = -1LL;
    if (!IsDynamicRank(inputs[0])) {
      if (inputs[0].size() == 0) {
        ret_size = 1;
      } else {
        ret_size = inputs[0].size();
      }
    }
    return {ret_size};
  });

REG_FALLBACK_BUILDER("NonZeroExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto input_shape = input->shape();
  if (IsDynamicRank(input_shape)) {
    MS_EXCEPTION(ValueError)
      << "For `NonZeroExt` op, would use unstack op, the dynamic rank is not support on input be empty.";
  }
  auto shape = ib->ShapeCalc(g_nonzero_ext_shapecalc, {input})[0];
  auto expanded_input = ib->Reshape(input, shape);
  auto output_tensor = ib->Emit("NonZero", {expanded_input});
  auto output_tuple = ib->Emit("Unstack", {output_tensor}, {{"axis", MakeValue<int64_t>(1LL)}});
  return {output_tuple};
});

class SoftmaxShapeCalc : public ShapeCalcFunctor {
 public:
  SoftmaxShapeCalc() : ShapeCalcFunctor("ShapeCalc_Softmax") {}
  ~SoftmaxShapeCalc() override = default;
  MS_DECLARE_PARENT(SoftmaxShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {dout_shape, dim}
    auto dim = inputs.at(1)[0];
    return {GetTransposeAxis(inputs.at(0), dim)};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t dout_rank = IsDynamicRank(inputs.at(0)) ? -1 : SizeToLong(inputs.at(0).size());
    return {dout_rank};
  }
};
REG_FUNCTOR("ShapeCalc_Softmax", SoftmaxShapeCalc);

REG_FALLBACK_BUILDER("SoftmaxBackward").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);

  auto shp = out->shape();
  auto dim_value_ptr = dim->BuildValue();
  int64_t dim_value{0};
  bool success = false;
  if (!(dim_value_ptr->isa<ValueAny>() || dim_value_ptr->isa<None>())) {
    dim_value = GetValue<int64_t>(dim_value_ptr);
    success = true;
  }
  if (success && IsLastAxis(shp, dim_value)) {
    auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
    return {dx};
  }
  auto reverse_axis = (IsDynamicRank(shp) || !success)
                        ? ib->ShapeCalc(std::make_shared<SoftmaxShapeCalc>(), {dout, dim}, {1})[0]
                        : ib->Value(GetTransposeAxis(shp, dim_value));
  out = ib->Transpose(out, reverse_axis);
  dout = ib->Transpose(dout, reverse_axis);
  auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
  dx = ib->Transpose(dx, reverse_axis);
  return {dx};
});

DEF_PURE_SHAPE_CALC(g_log_softmax_ext_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(kIndex0);
    size_t ndim = x_shape.size();
    int64_t ret;
    if (ndim == 0 || ndim == 1 || ndim == 3) {
      ret = 0;
    } else {
      ret = 1;
    }
    return {{ret}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto shape_out = inputs.at(kIndex0);
    if (IsDynamicRank(shape_out)) {
      return {-1};
    }
    return {1};
  });

REG_FALLBACK_BUILDER("LogSoftmaxExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto dtype = ib->GetInput(kIndex2);
  if (!ib->GetDtype(dtype)->isa<TypeNone>()) {
    auto dtype_opt = GetScalarValue<int64_t>(dtype->BuildValue());
    MS_CHECK_VALUE(dtype_opt.has_value(), "For 'MeanExt', dtype must have valid value.");
    input = ib->Cast(input, TypeIdToType(static_cast<TypeId>(dtype_opt.value())));
  }
  if (ib->GetDtype(dim)->isa<TypeNone>()) {
    dim = ib->ShapeCalc(g_log_softmax_ext_shapecalc, {input})[0];
    dim = ib->TupleGetItem(dim, 0);
  }
  auto out = ib->Emit("LogSoftmax", {input, dim});
  return {out};
});

REG_FALLBACK_BUILDER("LayerNormGradExt").SetBody(BODYFUNC(ib) {
  auto dy = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto rstd = ib->GetInput(kIndex4);
  auto gamma = ib->GetInput(kIndex5);
  return {ib->Emit("LayerNormGradV3", {dy, x, rstd, mean, gamma})};
});

REG_FALLBACK_BUILDER("ResizeLinear1D").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto coordinate_transformation_mode = ib->GetInput(kIndex2);

  // fetch scales
  NodePtr scales_node{nullptr};
  auto x_shape = x->shape();
  auto size_value_ptr = size->BuildValue();
  MS_EXCEPTION_IF_NULL(size_value_ptr);
  if (IsValueKnown(size_value_ptr)) {
    auto size_value = GetValue<std::vector<int64_t>>(size_value_ptr);
    std::vector<float> scales_vec{static_cast<float>(size_value.at(kIndex0)) / static_cast<float>(x_shape.at(kIndex2))};
    scales_node = ib->Value(scales_vec);
  } else {
    MS_LOG(EXCEPTION) << "For ResizeLinear1D, size should be const.";
  }

  auto new_x = ib->ExpandDims(x, -2);
  auto out =
    ib->Emit("ResizeD", {new_x, size, scales_node, coordinate_transformation_mode}, {{"mode", MakeValue("linear")}});
  auto real_out = ib->Squeeze(out, MakeValue(std::vector<int64_t>{-2}));

  return {real_out};
});

REG_FALLBACK_BUILDER("ResizeLinear1DGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto coordinate_transformation_mode = ib->GetInput(kIndex2);

  auto new_dout = ib->ExpandDims(ib->Cast(dout, TypeId::kNumberTypeFloat32), -2);
  NodePtr align_corners{nullptr};
  auto value_ptr = coordinate_transformation_mode->BuildValue();
  if (IsValueKnown(value_ptr)) {
    auto mode = static_cast<CoordinateTransformMode>(GetValue<int64_t>(value_ptr));
    align_corners = mode == CoordinateTransformMode ::ALIGN_CORNERS ? ib->Value(true) : ib->Value(false);
  } else {
    MS_LOG(EXCEPTION) << "For ResizeLinear1DGrad, coordinate_transformation_mode should be const.";
  }

  auto dx = ib->Emit("ResizeBilinearGrad", {new_dout, x, align_corners, ib->BoolNot(align_corners)});

  auto dout_type = dout->dtype()->type_id();
  auto real_dx = ib->Squeeze(ib->Cast(dx, dout_type), MakeValue(std::vector<int64_t>{-2}));

  return {real_dx};
});

REG_FALLBACK_BUILDER("Scatter").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto src = ib->GetInput(kIndex3);
  auto reduce = ib->GetInput(kIndex4);
  auto dim_val = dim->BuildValue();
  auto reduce_val = reduce->BuildValue();
  if (!IsValueKnown(dim_val) || !IsValueKnown(reduce_val)) {
    MS_EXCEPTION(ValueError) << "For `TensorScatterElements` op, the `dim` and `reduce` must currently be a constant!";
  }
  auto idx_shape = ib->GetShape(index);
  if (IsShapeNone(idx_shape)) {
    return {input};
  }
  auto out = ib->Emit("TensorScatterElements", {input, index, src, dim, reduce});
  return {out};
});

REG_FALLBACK_BUILDER("ScatterValue").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto src = ib->GetInput(kIndex3);
  auto reduce = ib->GetInput(kIndex4);
  auto dim_val = dim->BuildValue();
  auto reduce_val = reduce->BuildValue();
  if (!IsValueKnown(dim_val) || !IsValueKnown(reduce_val)) {
    MS_EXCEPTION(ValueError) << "For `TensorScatterElements` op, the `dim` and `reduce` must currently be a constant!";
  }
  auto idx_shape = ib->GetShape(index);
  if (IsShapeNone(idx_shape)) {
    return {input};
  }
  NodePtr expand_shape = nullptr;
  if (IsDynamic(idx_shape)) {
    expand_shape = ib->Emit("Shape", {index});
  } else {
    expand_shape = ib->Value(idx_shape);
  }
  auto src_tensor = ib->Emit("BroadcastTo", {ib->ScalarToTensor(src, input->dtype()), expand_shape});
  auto out = ib->Emit("TensorScatterElements", {input, index, src_tensor, dim, reduce});
  return {out};
});

REG_FALLBACK_BUILDER("ClampTensor").SetBody(BODYFUNC(ib) {
  // clamp equation: output = minimum(maximum(x, min), max)
  auto x = ib->GetInput(kIndex0);
  auto min = ib->GetInput(kIndex1);
  auto max = ib->GetInput(kIndex2);

  auto min_type_none = ib->GetDtype(min)->isa<TypeNone>();
  auto max_type_none = ib->GetDtype(max)->isa<TypeNone>();

  auto output = x;
  if (!min_type_none) {
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(min)->type_id()) {
      min = ib->Cast(min, ib->GetDtype(x)->type_id());
    }
    output = ib->Maximum(output, min);
  }

  if (!max_type_none) {
    if (ib->GetDtype(x)->type_id() != ib->GetDtype(max)->type_id()) {
      max = ib->Cast(max, ib->GetDtype(x)->type_id());
    }
    output = ib->Minimum(output, max);
  }

  return {output};
});

REG_FALLBACK_BUILDER("ClampScalar").SetBody(BODYFUNC(ib) {
  // clamp equation: output = minimum(maximum(x, min), max)
  auto x = ib->GetInput(kIndex0);
  auto min = ib->GetInput(kIndex1);
  auto max = ib->GetInput(kIndex2);

  auto min_type_none = ib->GetDtype(min)->isa<TypeNone>();
  auto max_type_none = ib->GetDtype(max)->isa<TypeNone>();

  auto output = x;
  if (!min_type_none) {
    min = ib->ScalarToTensor(min, ib->GetDtype(x));
    output = ib->Maximum(output, min);
  }

  if (!max_type_none) {
    max = ib->ScalarToTensor(max, ib->GetDtype(x));
    output = ib->Minimum(output, max);
  }

  return {output};
});

NodePtr PaddingTupleToTensor(NodePtr paddings, FallbackIRBuilder *ib) {
  auto padding_value = paddings->BuildValue();
  auto padding_vec = CheckAndConvertUtils::CheckIntOrTupleInt("padding", padding_value, "pad");
  auto padding_tensor = ib->Tensor(padding_vec);
  return padding_tensor;
}

bool IsInputNeedExpand(NodePtr paddingsTensor, NodePtr inputTensor) {
  auto padding_shape = paddingsTensor->shape();
  auto input_x_shape = inputTensor->shape();
  constexpr int64_t kScaleNum = 2;
  return ((padding_shape[0] / kScaleNum) + 1) == SizeToLong(input_x_shape.size());
}

REG_FALLBACK_BUILDER("ConstantPadND").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto value_tensor = ib->ScalarToTensor(value);
  if (value->dtype() != input_x->dtype()) {
    value_tensor = ib->Cast(value_tensor, input_x->dtype());
  }
  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, value_tensor},
                      {{"mode", MakeValue<string>("constant")}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
});

NodePtrList PadExpanderBase(FallbackIRBuilder *ib, const string &mode) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex1);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3", {input_x, padding_tensor, ib->EmitValue(kNone)},
                      {{"mode", MakeValue<string>(mode)}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
}

REG_FALLBACK_BUILDER("ReflectionPad1D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReflectionPad2D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReflectionPad3D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReplicationPad1D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "edge"); });

REG_FALLBACK_BUILDER("ReplicationPad2D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "edge"); });

REG_FALLBACK_BUILDER("ReplicationPad3D").SetBody(BODYFUNC(ib) { return PadExpanderBase(ib, "edge"); });

NodePtrList PadGradExpanderBase(FallbackIRBuilder *ib, const string &mode) {
  auto input_x = ib->GetInput(kIndex0);
  auto padding = ib->GetInput(kIndex2);

  auto padding_tensor = PaddingTupleToTensor(padding, ib);
  bool is_expand = IsInputNeedExpand(padding_tensor, input_x);
  if (is_expand) {
    input_x = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(0)});
  }
  auto out = ib->Emit("PadV3Grad", {input_x, padding_tensor},
                      {{"mode", MakeValue<string>(mode)}, {"paddings_contiguous", MakeValue(true)}});
  if (is_expand) {
    out = ib->Squeeze(out, MakeValue(ShapeVector{0}));
  }
  return {out};
}

REG_FALLBACK_BUILDER("ReflectionPad1DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReflectionPad2DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReflectionPad3DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "reflect"); });

REG_FALLBACK_BUILDER("ReplicationPad1DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "edge"); });

REG_FALLBACK_BUILDER("ReplicationPad2DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "edge"); });

REG_FALLBACK_BUILDER("ReplicationPad3DGrad").SetBody(BODYFUNC(ib) { return PadGradExpanderBase(ib, "edge"); });

REG_FALLBACK_BUILDER("Embedding").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto weight = ib->GetInput(kIndex1);
  auto padding_idx = ib->GetInput(kIndex2);
  auto max_norm = ib->GetInput(kIndex3);
  auto norm_type = ib->GetInput(kIndex4);

  auto max_norm_value = max_norm->BuildValue();

  if (max_norm_value != nullptr && !max_norm_value->isa<None>()) {
    auto norm_type_value = norm_type->BuildValue();
    if (!IsValueKnown(max_norm_value) || !IsValueKnown(norm_type_value)) {
      MS_INTERNAL_EXCEPTION(ValueError) << "For `Embedding` op, max_norm and norm_type must be constant!";
    }

    auto max_norm_double = static_cast<double>(GetValue<float>(max_norm_value));
    auto norm_type_double = static_cast<double>(GetValue<float>(norm_type_value));

    if (max_norm_double < 0) {
      MS_EXCEPTION(ValueError) << "For Embedding, the max_norm must be greater equal than 0, but got: "
                               << max_norm_double << ".";
    }

    // do EmbeddingRenorm
    auto new_input = ib->Emit(ops::kNameReshape, {input, ib->Value(std::vector<int64_t>{-1})});
    auto gather_out = ib->Emit(ops::kNameGather, {weight, new_input, ib->Value((int64_t)0), ib->Value((int64_t)0)});
    auto renorm_out = ib->Emit(ops::kNameRenorm, {gather_out},
                               {{"p", MakeValue<float>(norm_type_double)},
                                {"dim", MakeValue<int64_t>(0)},
                                {"maxnorm", MakeValue<float>(max_norm_double)}});

    if (IsDynamic(input->shape())) {
      MS_INTERNAL_EXCEPTION(ValueError)
        << "For `Embedding` op, dynamic_shape is not support on Fallback path, but got input shape: " << input->shape()
        << ".";
    }

    auto indices_size = SizeOf(input->shape());
    constexpr int64_t kMaxRangeSize = 1000000;
    auto indices = ib->Emit(ops::kNameRange, {ib->Value((int64_t)0), ib->Value(static_cast<int64_t>(indices_size)),
                                              ib->Value((int64_t)1), ib->Value(kMaxRangeSize)});
    auto gather_out2 = ib->Emit(ops::kNameGather, {renorm_out, indices, ib->Value((int64_t)0), ib->Value((int64_t)0)});
    auto mul_out = ib->Emit(ops::kNameMul, {gather_out, gather_out2});
    weight = ib->Emit(ops::kNameScatterUpdate, {weight, new_input, mul_out});
  }

  auto out = ib->Emit(ops::kNameGather, {weight, input, ib->Value((int64_t)0), ib->Value((int64_t)0)});
  return {out};
});

REG_FALLBACK_BUILDER("BatchNormExt").SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto weight = ib->GetInput(kIndex1);
  auto bias = ib->GetInput(kIndex2);
  auto running_mean = ib->GetInput(kIndex3);
  auto running_var = ib->GetInput(kIndex4);
  auto training = ib->GetInput(kIndex5);
  auto momentum = ib->GetInput(kIndex6);
  auto eps = ib->GetInput(kIndex7);
  auto format = ib->EmitValue(MakeValue<int64_t>(Format::NCHW));

  auto eps_ptr = eps->BuildValue();
  auto training_ptr = training->BuildValue();
  auto momentum_ptr = momentum->BuildValue();
  if (!IsValueKnown(eps_ptr) || !IsValueKnown(training_ptr) || !IsValueKnown(momentum_ptr)) {
    MS_EXCEPTION(ValueError) << "For `BatchNormExt` op, the `momentum` , `training` and `eps` must be a constant!";
  }
  auto eps_value = GetValue<float>(eps_ptr);
  auto training_value = GetValue<bool>(training_ptr);
  auto momentum_value = GetValue<float>(momentum_ptr);

  NodePtrList res{};
  auto bn_update_outputs = ib->Emit(prim::kPrimBNTrainingReduce->name(), {input, format}, {});
  auto sum = ib->TupleGetItem(bn_update_outputs, 0);
  auto square_sum = ib->TupleGetItem(bn_update_outputs, 1);

  if (training_value) {
    auto bn_training_outputs = ib->Emit(prim::kPrimBNTrainingUpdate->name(),
                                        {input, sum, square_sum, weight, bias, running_mean, running_var, format},
                                        {{"factor", MakeValue(momentum_value)}, {"epsilon", MakeValue(eps_value)}});
    (void)res.emplace_back(ib->TupleGetItem(bn_training_outputs, 0));
    (void)res.emplace_back(ib->TupleGetItem(bn_training_outputs, 3));
    (void)res.emplace_back(ib->TupleGetItem(bn_training_outputs, 4));
  } else {
    auto bn_infer_outputs = ib->Emit(prim::kPrimBNInfer->name(), {input, weight, bias, running_mean, running_var},
                                     {{"epsilon", MakeValue(eps_value)}});
    (void)res.emplace_back(std::move(bn_infer_outputs));
    (void)res.emplace_back(sum);
    (void)res.emplace_back(square_sum);
  }
  return {ib->MakeTuple(res)};
});

REG_FALLBACK_BUILDER("BatchNormGradExt").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto input = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto running_var = ib->GetInput(kIndex4);
  auto saved_mean = ib->GetInput(kIndex5);
  auto saved_rstd = ib->GetInput(kIndex6);
  auto training = ib->GetInput(kIndex7);
  auto eps = ib->GetInput(kIndex8);

  auto eps_ptr = eps->BuildValue();
  auto training_ptr = training->BuildValue();

  if (!IsValueKnown(eps_ptr) || !IsValueKnown(training_ptr)) {
    MS_EXCEPTION(ValueError) << "For `BatchNormGradExt` op, the  `training` and `eps` must be a constant!";
  }
  auto eps_value = GetValue<float>(eps_ptr);
  auto training_value = GetValue<bool>(training_ptr);

  NodePtrList res{};
  auto bn_update_grad_outputs = ib->Emit(prim::kPrimBNTrainingUpdateGrad->name(), {dout, input, saved_mean, saved_rstd},
                                         {{"epsilon", MakeValue(eps_value)}});
  auto diff_scale = ib->TupleGetItem(bn_update_grad_outputs, 0);
  auto diff_offset = ib->TupleGetItem(bn_update_grad_outputs, 1);

  if (training_value) {
    auto bn_reduce_grad_outputs = ib->Emit(prim::kPrimBNTrainingReduceGrad->name(),
                                           {dout, input, diff_scale, diff_offset, weight, saved_mean, saved_rstd},
                                           {{"epsilon", MakeValue(eps_value)}});
    (void)res.emplace_back(std::move(bn_reduce_grad_outputs));
    (void)res.emplace_back(diff_scale);
    (void)res.emplace_back(diff_offset);
  } else {
    auto bn_infer_grad_outputs =
      ib->Emit(prim::kPrimBNInferGrad->name(), {dout, weight, running_var}, {{"epsilon", MakeValue(eps_value)}});
    (void)res.emplace_back(std::move(bn_infer_grad_outputs));
    (void)res.emplace_back(diff_scale);
    (void)res.emplace_back(diff_offset);
  }
  return {ib->MakeTuple(res)};
});

REG_FALLBACK_BUILDER("GroupNorm").SetBody(BODYFUNC(ib) {
  constexpr const size_t kNumberTwo = 2;
  auto input = ib->GetInput(kIndex0);
  auto x = ib->Cast(input, kFloat32);
  auto groups = ib->GetInput(kIndex1);
  auto weight = ib->Cast(ib->GetInput(kIndex2), kFloat32);
  auto bias = ib->Cast(ib->GetInput(kIndex3), kFloat32);
  auto eps = ib->GetInput(kIndex4);
  if (!IsValueKnown(eps->BuildValue()) || !IsValueKnown(groups->BuildValue())) {
    MS_EXCEPTION(ValueError) << "For `GroupNorm` op, the  `num_groups` and `eps` must be a constant!";
  }
  auto eps_value = ib->Tensor(GetValue<float>(eps->BuildValue()));
  auto num_groups = GetValue<int64_t>(groups->BuildValue());

  if (IsDynamic(input->shape())) {
    MS_INTERNAL_EXCEPTION(ValueError)
      << "For `GroupNorm` op, dynamic_shape is not support on Fallback path, but got input shape: " << input->shape()
      << ".";
  }

  auto x_shape = x->shape();
  const int64_t batch = x_shape[0];
  const int64_t channel = x_shape[1];
  const int64_t HxW = (x_shape.size() == kNumberTwo)
                        ? 1
                        : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>());
  const int64_t g = channel / num_groups;

  auto x_reshape = ib->Reshape(x, ShapeVector{batch, num_groups, g * HxW});
  ShapeVector weight_and_bias_reshape(x_shape.size() - 1, 1);
  weight_and_bias_reshape[0] = channel;
  auto weight_reshape = ib->Reshape(weight, weight_and_bias_reshape);
  auto bias_reshape = ib->Reshape(bias, weight_and_bias_reshape);
  auto factor = ib->Tensor(HxW * g, kFloat32);

  auto mean = ib->Emit("ReduceMean", {x_reshape, ib->Value<std::vector<int64_t>>({kNumberTwo}), ib->Value(true)});
  auto variance = ib->Div(ib->ReduceSum(ib->Square(ib->Sub(x_reshape, mean)), ShapeVector{kNumberTwo}, true), factor);
  auto rstd = ib->Reciprocal(ib->Sqrt(ib->Add(variance, eps_value)));
  auto tmp1 = ib->Reshape(ib->Mul(ib->Sub(x_reshape, mean), rstd), x_shape);
  auto output = ib->Cast(ib->Add(ib->Mul(tmp1, weight_reshape), bias_reshape), input->dtype());
  auto mean_out = ib->Cast(ib->Reshape(mean, ShapeVector{batch, num_groups}), input->dtype());
  auto rstd_out = ib->Cast(ib->Reshape(rstd, ShapeVector{batch, num_groups}), input->dtype());
  return {ib->MakeTuple({output, mean_out, rstd_out})};
});

REG_FALLBACK_BUILDER("GroupNormGrad").SetBody(BODYFUNC(ib) {
  constexpr const size_t kNumber2 = 2;
  constexpr float kFloatThree = 3.0;
  auto dy = ib->Cast(ib->GetInput(kIndex0), kFloat32);
  auto input = ib->GetInput(kIndex1);
  auto x = ib->Cast(input, kFloat32);
  auto mean = ib->Cast(ib->GetInput(kIndex2), kFloat32);
  auto rstd = ib->Cast(ib->GetInput(kIndex3), kFloat32);
  auto gamma = ib->Cast(ib->GetInput(kIndex4), kFloat32);
  auto groups = ib->GetInput(kIndex5);

  if (!IsValueKnown(groups->BuildValue())) {
    MS_EXCEPTION(ValueError) << "For `GroupNormGrad` op, the  `num_groups` must be a constant!";
  }
  auto x_shape = x->shape();

  if (IsDynamic(x_shape)) {
    MS_INTERNAL_EXCEPTION(ValueError)
      << "For `GroupNormGrad` op, dynamic_shape is not support on Fallback path, but got input shape: "
      << input->shape() << ".";
  }

  auto num_groups = GetValue<int64_t>(groups->BuildValue());
  const int64_t batch = x_shape[0];
  const int64_t channel = x_shape[1];
  const int64_t HxW = (x_shape.size() == kNumber2)
                        ? 1
                        : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>());
  const int64_t g = channel / num_groups;
  auto ds = ib->ReduceSum(ib->Reshape(ib->Mul(dy, x), ShapeVector{batch, channel, HxW}), ShapeVector{kNumber2});
  auto db = ib->ReduceSum(ib->Reshape(dy, ShapeVector{batch, channel, HxW}), ShapeVector{kNumber2});

  auto ds_reshape = ib->Reshape(ds, ShapeVector{batch, num_groups, g});
  auto db_reshape = ib->Reshape(db, ShapeVector{batch, num_groups, g});
  auto mean_reshape = ib->Reshape(mean, ShapeVector{batch, num_groups, 1});
  auto rstd_reshape = ib->Reshape(rstd, ShapeVector{batch, num_groups, 1});
  auto dy_reshape = ib->Reshape(dy, ShapeVector{batch, num_groups, g, HxW});
  auto x_reshape = ib->Reshape(x, ShapeVector{batch, num_groups, g, HxW});

  auto three = ib->Tensor(kFloatThree, kFloat32);
  auto factor = ib->Tensor(HxW * g, kFloat32);

  auto ds_val = ib->ReduceSum(
    ib->Reshape(ib->Mul(ds, ib->Reshape(gamma, ShapeVector{1, channel})), ShapeVector{batch, num_groups, g}),
    ShapeVector{kNumber2});
  auto db_val = ib->ReduceSum(
    ib->Reshape(ib->Mul(db, ib->Reshape(gamma, ShapeVector{1, channel})), ShapeVector{batch, num_groups, g}),
    ShapeVector{kNumber2});

  auto tmp1 = ib->Mul(rstd_reshape, ib->Reshape(gamma, ShapeVector{1, num_groups, g}));
  auto tmp2 = ib->Div(ib->Mul(ib->Sub(ib->Mul(db_val, mean), ds_val), ib->Pow(rstd, three)), factor);
  auto tmp3 = ib->Neg(ib->Add(ib->Mul(tmp2, mean), ib->Div(ib->Mul(db_val, rstd), factor)));
  auto tmp1_reshape = ib->Reshape(tmp1, ShapeVector{batch, num_groups, g, 1});
  auto tmp2_reshape = ib->Reshape(tmp2, ShapeVector{batch, num_groups, 1, 1});
  auto tmp3_reshape = ib->Reshape(tmp3, ShapeVector{batch, num_groups, 1, 1});

  auto dx = ib->Cast(
    ib->Reshape(ib->Add(ib->Add(ib->Mul(dy_reshape, tmp1_reshape), ib->Mul(x_reshape, tmp2_reshape)), tmp3_reshape),
                x_shape),
    input->dtype());
  auto dgamma = ib->Cast(
    ib->Reshape(
      ib->ReduceSum(ib->Mul(ib->Sub(ds_reshape, ib->Mul(db_reshape, mean_reshape)), rstd_reshape), ShapeVector{0}),
      ShapeVector{channel}),
    input->dtype());
  auto dbeta = ib->Cast(ib->ReduceSum(db, ShapeVector{0}), input->dtype());
  return {ib->MakeTuple({dx, dgamma, dbeta})};
});

REG_FALLBACK_BUILDER("Index").SetBody(BODYFUNC(ib) {
  auto input_tensor = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto input_shape = input_tensor->shape();
  auto input_shape_nums = input_shape.size();
  if (input_shape_nums == 0) {
    MS_EXCEPTION(ValueError) << "For `Index` op, too many indices for tensor of dimension " << input_shape_nums;
  }
  auto indices_abs = indices->abstract();
  MS_EXCEPTION_IF_NULL(indices_abs);
  auto indices_shape = indices_abs->GetShape();
  MS_EXCEPTION_IF_NULL(indices_shape);
  auto indices_is_dyn_seq = indices_shape->isa<abstract::DynamicSequenceShape>();
  if (indices_is_dyn_seq) {
    MS_EXCEPTION(ValueError) << "For `Index` op, 'indices' shape can not DynamicSequenceShape.";
  }
  std::vector<ShapeVector> indices_shapes = indices->shapes();
  if (indices_shapes.empty()) {
    MS_EXCEPTION(ValueError) << "For 'Index', 'indices' shape can't be empty.";
  }
  auto indices_nums = indices_shapes.size();
  if (!IsDynamicRank(input_shape) && indices_nums > input_shape_nums) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << input_shape_nums << " (got "
                             << indices_nums << ")";
  }

  bool needCast = false;
  TypeId indicesDtype = ib->TupleGetItem(indices, kIndex0)->dtype()->type_id();
  // Expand indices, then insert input and value
  NodePtrList new_indices;
  new_indices.emplace_back(input_tensor);
  for (size_t i = 0; i < indices_nums; ++i) {
    auto tensor = ib->TupleGetItem(indices, i);
    auto type_id = tensor->dtype()->type_id();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError) << "For 'Index', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      auto shape = tensor->shape();
      auto rank = SizeToLong(shape.size());
      if (IsDynamicRank(shape)) {
        MS_EXCEPTION(ValueError) << "For 'Index', when the dytpe of tensor is bool or uint8 ,it would use "
                                 << "'Unstack' op, the dynamic rank is not support";
      }
      for (int64_t j = 0; j < rank; j++) {
        auto srcIdx = new_indices.size() - 1 + j;
        if (!IsDynamic(shape) && !IsDynamic(input_shape) && shape[j] != input_shape[srcIdx]) {
          MS_EXCEPTION(ValueError) << "For 'Index', the shape of the mask " << shape << " at index " << j
                                   << " does not match the shape of the indexed tensor " << input_shape << " at index "
                                   << srcIdx;
        }
      }
      // The InnerNonZero(input) output shape is (rank) * (non zero number)
      auto nonzero_tensor = ib->Emit("InnerNonZero", {tensor});
      auto unstack_tensor = ib->Emit("Unstack", {nonzero_tensor}, {{"axis", MakeValue<int64_t>(0LL)}});
      // The nonzero and unstack will generation tuple[tensor], the tuple size is input's rank
      for (int64_t j = 0; j < rank; j++) {
        new_indices.emplace_back(ib->TupleGetItem(unstack_tensor, j));
      }
    } else {
      new_indices.emplace_back(tensor);
    }
    if (indicesDtype != type_id) {
      needCast = true;
    }
  }
  if (needCast) {
    for (size_t i = 1; i < new_indices.size(); i++) {
      if (new_indices[i]->dtype()->type_id() == kNumberTypeInt32) {
        new_indices[i] = ib->Cast(new_indices[i], kInt64);
      }
    }
  }
  auto output = ib->Emit("InnerIndex", new_indices);
  return {output};
});

REG_FALLBACK_BUILDER("InplaceIndexPut").SetBody(BODYFUNC(ib) {
  auto input_tensor = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto values_tensor = ib->GetInput(kIndex2);
  auto accumulate = ib->GetInput(kIndex3);

  auto indices_abs = indices->abstract();
  MS_EXCEPTION_IF_NULL(indices_abs);
  auto indices_shape = indices_abs->GetShape();
  MS_EXCEPTION_IF_NULL(indices_shape);
  auto indices_is_dyn_seq = indices_shape->isa<abstract::DynamicSequenceShape>();
  if (indices_is_dyn_seq) {
    MS_EXCEPTION(ValueError) << "For `InplaceIndexPut` op, 'indices' shape can not DynamicSequenceShape.";
  }
  std::vector<ShapeVector> indices_shapes = indices->shapes();
  if (indices_shapes.empty()) {
    return {input_tensor};
  }
  auto indices_nums = indices_shapes.size();
  auto input_shape = ib->GetShape(input_tensor);
  auto value_shape = ib->GetShape(values_tensor);
  if (!IsDynamic(input_shape)) {
    if (!IsDynamic(value_shape)) {
      auto input_numel = std::accumulate(input_shape.begin(), input_shape.end(), kIndex1, std::multiplies<int64_t>());
      auto values_numel = std::accumulate(value_shape.begin(), value_shape.end(), kIndex1, std::multiplies<int64_t>());
      if (input_numel == 0 || values_numel == 0 || indices_nums == 0) {
        return {input_tensor};
      }
    }
    auto input_shape_nums = input_shape.size();
    if (input_shape_nums == 0) {
      MS_EXCEPTION(ValueError) << "For `InplaceIndexPut` op, too many indices for tensor of dimension "
                               << input_shape_nums;
    }
  }
  if (!IsDynamicRank(input_shape) && indices_nums > input_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension " << input_shape.size()
                             << " (got " << indices_nums << ")";
  }
  // Expand indices, then insert input and value
  bool needCast = false;
  TypeId indicesDtype = ib->TupleGetItem(indices, kIndex0)->dtype()->type_id();
  NodePtrList new_indices;
  new_indices.emplace_back(input_tensor);
  for (size_t i = 0; i < indices_nums; ++i) {
    auto tensor = ib->TupleGetItem(indices, i);
    auto type_id = tensor->dtype()->type_id();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError)
        << "For 'InplaceIndexPut', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      auto shape = tensor->shape();
      auto rank = SizeToLong(shape.size());
      if (IsDynamicRank(shape)) {
        MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', when the dytpe of tensor is bool or uint8 ,it would use "
                                 << "'Unstack' op, the dynamic rank is not support";
      }
      for (int64_t j = 0; j < rank; j++) {
        auto srcIdx = new_indices.size() - 1 + j;
        if (!IsDynamic(shape) && !IsDynamic(input_shape) && shape[j] != input_shape[srcIdx]) {
          MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', the shape of the mask " << shape << " at index " << j
                                   << " does not match the shape of the indexed tensor " << input_shape << " at index "
                                   << srcIdx;
        }
      }
      // For aclnnIndexPutImpl op, the indices element dtype supports bool.
      if (type_id == kNumberTypeUInt8) {
        // The InnerNonZero(input) output shape is (rank) * (non zero number)
        auto nonzero_tensor = ib->Emit("InnerNonZero", {tensor});
        auto unstack_tensor = ib->Emit("Unstack", {nonzero_tensor}, {{"axis", MakeValue<int64_t>(0LL)}});
        // The nonzero and unstack will generation tuple[tensor], the tuple size is input's rank
        for (int64_t j = 0; j < rank; j++) {
          new_indices.emplace_back(ib->TupleGetItem(unstack_tensor, j));
        }
      } else {
        new_indices.emplace_back(tensor);
      }
    } else {
      new_indices.emplace_back(tensor);
    }
    if (indicesDtype != type_id) {
      needCast = true;
    }
  }
  if (needCast) {
    for (size_t i = 1; i < new_indices.size(); i++) {
      if (new_indices[i]->dtype()->type_id() == kNumberTypeInt32) {
        new_indices[i] = ib->Cast(new_indices[i], kInt64);
      }
    }
  }
  new_indices.emplace_back(values_tensor);
  new_indices.emplace_back(accumulate);
  auto output = ib->Emit("InnerInplaceIndexPut", new_indices);
  return {output};
});
}  // namespace expander
}  // namespace mindspore
