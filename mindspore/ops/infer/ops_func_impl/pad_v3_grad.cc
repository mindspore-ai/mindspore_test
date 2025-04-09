/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/pad_v3_grad.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore {
namespace ops {
namespace {
void PaddingsValueCheck(const PrimitivePtr &primitive, const ShapeVector &dout_shape,
                        const std::vector<int64_t> &paddings_val, Mode mode, const std::string &prim_name) {
  const int64_t max_x_dim = 5;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    (void)CheckAndConvertUtils::CheckInteger("x_dim", SizeToLong(dout_shape.size()), kLessThan, max_x_dim, prim_name);
    // For Ascend, ge::PadV3Grad only support paddings has positive value, and this node is called when mode
    // is not 'constant'
    if (mode != Mode::CONSTANT) {
      (void)CheckAndConvertUtils::CheckPositiveVector("paddings", paddings_val, prim_name);
    }
  }
}

std::optional<std::vector<int64_t>> PadV3GradFetchPaddingsValFromArg(const PrimitivePtr &primitive,
                                                                     const AbstractBasePtr &paddings_arg) {
  std::vector<int64_t> paddings_val;
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  auto padding_type = paddings_arg->GetType();
  if (padding_type->isa<TensorType>()) {
    auto paddings_shape_ptr = paddings_arg->GetShape();
    MS_EXCEPTION_IF_NULL(paddings_shape_ptr);
    auto paddings_value_ptr = paddings_arg->GetValue();
    MS_EXCEPTION_IF_NULL(paddings_value_ptr);
    if (paddings_shape_ptr->IsDynamic() || paddings_value_ptr->isa<ValueAny>()) {
      return std::nullopt;
    }
    paddings_val =
      CheckAndConvertUtils::CheckTensorIntValue("paddings value", paddings_value_ptr, prim_name, padding_type);
  } else if (padding_type->isa<Tuple>() || padding_type->isa<List>()) {
    auto value_ptr = paddings_arg->GetValue();
    if (IsValueKnown(value_ptr)) {
      paddings_val = CheckAndConvertUtils::CheckIntOrTupleInt("paddings value", paddings_arg, prim_name);
    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
  }
  return paddings_val;
}

std::vector<int64_t> PadV3GradDealWithPaddingValue(const PrimitivePtr &primitive,
                                                   const std::vector<int64_t> &dout_shape, size_t paddings_size,
                                                   const std::vector<int64_t> &ori_paddings_val, Mode mode,
                                                   bool paddings_contiguous) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> paddings_val(ori_paddings_val);
  if (paddings_size == dout_shape.size() * kIndex2) {
    MS_LOG(INFO) << "For " << prim_name
                 << ", the paddings' val has been changed in ascend backend pass, which causes paddings' size expanded "
                    "to 2 times x_rank(8, or 10)";
    // (0, 1, 2, 3, 4, 5, 6, 7) -> (6, 7, 4, 5, 2, 3, 0, 1)
    std::reverse(paddings_val.begin(), paddings_val.end());
    for (size_t i = 1; i < paddings_val.size(); i += kIndex2) {
      std::swap(paddings_val[i - 1], paddings_val[i]);
    }
  } else {
    PaddingsValueCheck(primitive, dout_shape, paddings_val, mode, prim_name);
    if (!paddings_contiguous) {
      for (size_t i = 0; i < paddings_size; ++i) {
        if (i % kIndex2 == 0) {
          paddings_val[i] = ori_paddings_val[i / kIndex2];
        } else {
          paddings_val[i] = ori_paddings_val[(i + paddings_size) / kIndex2];
        }
      }
    }
  }
  return paddings_val;
}

std::vector<int64_t> PadV3GradInferOutputShapeWithPaddings(const PrimitivePtr &primitive,
                                                           const std::vector<int64_t> &dout_shape, size_t paddings_size,
                                                           const std::vector<int64_t> &paddings_val) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> out_shape;
  if (paddings_size == kIndex2) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 2", SizeToLong(kIndex3), kEqual,
                                             SizeToLong(dout_shape.size()), prim_name);
    (void)out_shape.emplace_back(dout_shape[0]);
    (void)out_shape.emplace_back(dout_shape[1]);
    (void)out_shape.emplace_back(dout_shape[kIndex2] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kIndex4) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 4", SizeToLong(kIndex4), kEqual,
                                             SizeToLong(dout_shape.size()), prim_name);
    (void)out_shape.emplace_back(dout_shape[0]);
    (void)out_shape.emplace_back(dout_shape[1]);
    (void)out_shape.emplace_back(dout_shape[kIndex2] - paddings_val[kIndex2] - paddings_val[kIndex3]);
    (void)out_shape.emplace_back(dout_shape[kIndex3] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kIndex6) {
    (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 6", SizeToLong(kIndex5), kEqual,
                                             SizeToLong(dout_shape.size()), prim_name);
    (void)out_shape.emplace_back(dout_shape[0]);
    (void)out_shape.emplace_back(dout_shape[1]);
    (void)out_shape.emplace_back(dout_shape[kIndex2] - paddings_val[kIndex4] - paddings_val[kIndex5]);
    (void)out_shape.emplace_back(dout_shape[kIndex3] - paddings_val[kIndex2] - paddings_val[kIndex3]);
    (void)out_shape.emplace_back(dout_shape[kIndex4] - paddings_val[0] - paddings_val[1]);
  } else if (paddings_size == kIndex2 * dout_shape.size()) {
    MS_LOG(INFO) << "For " << prim_name
                 << ", the paddings' val has been changed in ascend backend pass, which causes paddings' size expanded "
                    "to 2 times x_rank(8, or 10)";
    MS_ASSERT(paddings_val.size() == kIndex8 || paddings_val.size() == kIndex10);
    for (size_t i = 0; i < dout_shape.size(); i++) {
      (void)out_shape.push_back(dout_shape[i] - paddings_val[paddings_size - kIndex2 * i - kIndex1] -
                                paddings_val[paddings_size - kIndex2 * (i + kIndex1)]);
    }
  } else {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the length of paddings must be 2, 4 or 6, but got "
                             << paddings_size;
  }
  (void)CheckAndConvertUtils::CheckPositiveVector("out_shape", out_shape, prim_name);
  return out_shape;
}

abstract::ShapePtr PadV3GradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = primitive->name();
  auto dout_shape_ptr = input_args[kIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(dout_shape_ptr);
  // support dynamic rank
  if (dout_shape_ptr->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  const auto &dout_shape = dout_shape_ptr->GetShapeVector();
  auto ori_paddings_val_opt = PadV3GradFetchPaddingsValFromArg(primitive, input_args[kIndex1]);
  if (dout_shape_ptr->IsDynamic() || !ori_paddings_val_opt.has_value()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(dout_shape.size(), abstract::Shape::kShapeDimAny));
  }

  auto ori_paddings_val = ori_paddings_val_opt.value();
  auto paddings_size = ori_paddings_val.size();

  auto mode_opt = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  auto paddings_contiguous_opt = GetScalarValue<bool>(input_args[kIndex3]->GetValue());
  if (!mode_opt.has_value() || !paddings_contiguous_opt.has_value()) {
    MS_EXCEPTION(RuntimeError) << "For " << prim_name << ", `mode` and `paddings_contiguous` should be const.";
  }
  auto mode = static_cast<Mode>(mode_opt.value());
  auto paddings_contiguous = paddings_contiguous_opt.value();
  auto paddings_val =
    PadV3GradDealWithPaddingValue(primitive, dout_shape, paddings_size, ori_paddings_val, mode, paddings_contiguous);
  auto out_shape = PadV3GradInferOutputShapeWithPaddings(primitive, dout_shape, paddings_size, paddings_val);

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadV3GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &prin_name = primitive->name();
  std::map<std::string, TypePtr> args = {{"x", input_args[kIndex0]->GetType()}};
  auto mode_opt = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  if (!mode_opt.has_value()) {
    MS_EXCEPTION(RuntimeError) << "For " << prin_name << ", `mode` should be const.";
  }
  auto mode = static_cast<Mode>(mode_opt.value());
  if (mode == Mode::CONSTANT) {
    return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                     {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBool},
                                                     prin_name);
  } else {
    return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                     {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64,
                                                      kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
                                                     prin_name);
  }
}
}  // namespace

BaseShapePtr PadV3GradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return PadV3GradInferShape(primitive, input_args);
}

TypePtr PadV3GradFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return PadV3GradInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore