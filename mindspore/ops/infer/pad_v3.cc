/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ops_utils/op_constants.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/op_name.h"
#include "infer/pad_v3.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto nTwo = 2;
constexpr auto kPaddingsSizeTwo = 2;
constexpr auto kPaddingsSizeFour = 4;
constexpr auto kConstantInput = 3;
void PaddingsSizeCheck(const PrimitivePtr &primitive, const int64_t paddings_size, const int64_t size) {
  constexpr int64_t kPaddingsSizeSix = 6;
  constexpr int64_t nThree = 3;
  constexpr int64_t nFour = 4;
  constexpr int64_t nFive = 5;
  auto prim_name = primitive->name();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto is_dyn_paddings = primitive->GetAttr("is_dyn_paddings");
    if (is_dyn_paddings != nullptr && GetValue<bool>(is_dyn_paddings)) {
      if (paddings_size / nTwo != size) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', paddings length must be equal to " << size * nTwo;
      }
      return;
    }
  }
  auto mode = GetValue<std::string>(primitive->GetAttr("mode"));
  if (mode == kConstant) {
    if (paddings_size / nTwo > size) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "' constant mode, paddings length too large for input dims, the pad dims must be less than or equal to "
        << size;
    }
    if (paddings_size % nTwo == 1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "' constant mode, paddings length must be divisible by 2";
    }
  } else {
    if (paddings_size == kPaddingsSizeTwo) {
      (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 2", size, kEqual, nThree,
                                               prim_name);
    } else if (paddings_size == kPaddingsSizeFour) {
      (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 4", size, kEqual, nFour,
                                               prim_name);
    } else if (paddings_size == kPaddingsSizeSix) {
      (void)CheckAndConvertUtils::CheckInteger("input dims when padding's size equal 6", size, kEqual, nFive,
                                               prim_name);
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the length of paddings must be 2, 4 or 6, but got "
                               << paddings_size;
    }
  }
}

void PaddingsValueCheck(const std::string &prim_name, const std::vector<int64_t> &x_shape,
                        const std::vector<int64_t> &paddings_val) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    (void)CheckAndConvertUtils::CheckPositiveVector("paddings", paddings_val, prim_name);
  }
  auto x_shape_reverse = x_shape;
  std::reverse_copy(x_shape.begin(), x_shape.end(), x_shape_reverse.begin());
  for (size_t i = 0; i < paddings_val.size(); i++) {
    if (paddings_val[i] < 0) {
      (void)CheckAndConvertUtils::CheckInteger("paddings_value", paddings_val[i], CompareEnum::kGreaterEqual,
                                               -x_shape_reverse[i / nTwo], prim_name);
    }
  }
}

void ReflectModeCheck(const std::string &prim_name, const int64_t paddings_size, std::vector<int64_t> x_shape,
                      std::vector<int64_t> paddings_val, const int64_t size, bool is_paddings_changed) {
  if (is_paddings_changed) {
    return;
  }

  constexpr int64_t kReflectMaxDims = 4;
  (void)CheckAndConvertUtils::CheckInteger("input dims for reflect mode", size, kLessEqual, kReflectMaxDims, prim_name);
  if (paddings_size == kPaddingsSizeTwo) {
    if (paddings_val[0] >= x_shape[kInputIndex2] || paddings_val[1] >= x_shape[kInputIndex2]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "' reflect mode, Padding size must be less than the corresponding input dimension"
                               << ", but got: padding (" << paddings_val[0] << ',' << paddings_val[1]
                               << ") at dimension 2 of input:[" << x_shape[kInputIndex2] << "]";
    }
  }
  if (paddings_size == kPaddingsSizeFour) {
    if (paddings_val[0] >= x_shape[kInputIndex3] || paddings_val[1] >= x_shape[kInputIndex3]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "' reflect mode, Padding size must be less than the corresponding input dimension"
                               << ", but got: padding (" << paddings_val[0] << ',' << paddings_val[1]
                               << ") at dimension 3 of input:[" << x_shape[kInputIndex3] << "]";
    }
    if (paddings_val[kInputIndex2] >= x_shape[kInputIndex2] || paddings_val[kInputIndex3] >= x_shape[kInputIndex2]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "' reflect mode, Padding size must be less than the corresponding input dimension"
                               << ", but got: padding (" << paddings_val[kInputIndex2] << ','
                               << paddings_val[kInputIndex3] << ") at dimension 2 of input:[" << x_shape[kInputIndex2]
                               << "]";
    }
  }
}

abstract::ShapePtr PaddingNoTensor(abstract::BaseShapePtr paddings_shape_ptr, const std::vector<int64_t> &x_shape) {
  auto paddings_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(paddings_shape_ptr)[kShape];
  (void)CheckAndConvertUtils::CheckInteger("paddings_dim", SizeToLong(paddings_shape.size()), kEqual, kDim1, "PadV3");
  (void)CheckAndConvertUtils::CheckInteger("paddings_length", paddings_shape[kIndex0], kLessEqual,
                                           SizeToLong(x_shape.size() * nTwo), "PadV3");
  size_t pad_dim = 0;
  if (paddings_shape.size() >= 1) {
    pad_dim = paddings_shape[0] / nTwo;
  }
  auto out_shape = x_shape;
  auto dim_size = x_shape.size();
  for (size_t i = dim_size - pad_dim; i < dim_size; ++i) {
    out_shape[i] = abstract::Shape::kShapeDimAny;
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

void CheckAscendInputXDim(const size_t &x_dim, const std::string &prim_name) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice && x_dim > kDim5) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dimension of 'x' must be no more than " << kDim5
                             << " while running in Ascend.";
  }
}

void AscendTransformPaddingsAttr(const PrimitivePtr &primitive,
                                 std::vector<std::pair<int64_t, int64_t>> *ori_paddings_attr) {
  // If the `paddings` comes from the node added by pass, there are two features as followed:
  // 1. the length of `paddings` is twice than the rank of `x`.
  // 2. the mapper between `x` and `paddings` is lower to lower,
  //    which is different from that in another backends, which is lower to higher.
  // So, the transform should be activated only where the `paddings` is from the node added by pass.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto is_dyn_paddings = primitive->GetAttr("is_dyn_paddings");
    if (is_dyn_paddings != nullptr && GetValue<bool>(is_dyn_paddings)) {
      std::reverse(ori_paddings_attr->begin(), ori_paddings_attr->end());
    }
  }
}

std::vector<std::pair<int64_t, int64_t>> PadV3DealWithPaddings(const PrimitivePtr &primitive,
                                                               const std::vector<int64_t> &x_shape,
                                                               const std::vector<int64_t> &ori_paddings_val) {
  const auto &prim_name = primitive->name();
  auto is_paddings_changed{false};
  if (primitive->HasAttr("is_paddings_changed")) {
    MS_LOG(INFO) << "For " << prim_name
                 << ", the paddings' val has been changed in ascend backend pass, which causes paddings' size expanded "
                    "to 2 times x_rank";
    is_paddings_changed = GetValue<bool>(primitive->GetAttr("is_paddings_changed"));
  }

  auto x_rank = SizeToLong(x_shape.size());
  int64_t paddings_size = SizeToLong(ori_paddings_val.size());
  auto mode = GetValue<std::string>(primitive->GetAttr(kAttrMode));
  if (mode != kConstant) {
    constexpr int64_t kOtherMinDims = 3;
    (void)CheckAndConvertUtils::CheckInteger("input dims for edge, reflect or circular mode", x_rank, kGreaterEqual,
                                             kOtherMinDims, prim_name);
    if (mode == kReflect) {
      ReflectModeCheck(prim_name, paddings_size, x_shape, ori_paddings_val, x_rank, is_paddings_changed);
    } else {
      constexpr int64_t kEdgeMaxDims = 5;
      (void)CheckAndConvertUtils::CheckInteger("input dims for edge mode", x_rank, kLessEqual, kEdgeMaxDims, prim_name);
    }
  }

  std::vector<int64_t> paddings_val(ori_paddings_val);
  if (is_paddings_changed) {
    // (0, 1, 2, 3, 4, 5, 6, 7) -> (6, 7, 4, 5, 2, 3, 0, 1)
    std::reverse(paddings_val.begin(), paddings_val.end());
    for (size_t i = 1; i < paddings_val.size(); i += kInputIndex2) {
      std::swap(paddings_val[i - 1], paddings_val[i]);
    }
  } else {
    PaddingsSizeCheck(primitive, paddings_size, x_rank);
    // Checker: whether paddings_value + x_shape_value < 0 or not
    PaddingsValueCheck(prim_name, x_shape, ori_paddings_val);

    auto paddings_contiguous = GetValue<bool>(primitive->GetAttr("paddings_contiguous"));
    if (paddings_contiguous == false) {
      std::vector<int64_t> tmp = paddings_val;
      for (int64_t i = 0; i < paddings_size; ++i) {
        if (i % nTwo == 0) {
          paddings_val[LongToSize(i)] = tmp[LongToSize(i / nTwo)];
        } else {
          paddings_val[LongToSize(i)] = tmp[LongToSize((i + paddings_size) / nTwo)];
        }
      }
    }
  }

  std::vector<std::pair<int64_t, int64_t>> paddings_attr;
  for (int64_t i = 0; i < x_rank; ++i) {
    if (nTwo * i >= paddings_size) {
      paddings_attr.push_back(std::make_pair(int64_t(0), int64_t(0)));
    } else {
      paddings_attr.push_back(
        std::make_pair(paddings_val[LongToSize(nTwo * i)], paddings_val[LongToSize(nTwo * i + 1)]));
    }
  }
  AscendTransformPaddingsAttr(primitive, &paddings_attr);
  return paddings_attr;
}

abstract::ShapePtr PadV3InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  if (x_shape_ptr->IsDimUnknown()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  auto x_shape = x_shape_ptr->GetShapeVector();
  auto dim_size = x_shape.size();
  if (dim_size == 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dimension of 'x' must bigger than 0.";
  }
  CheckAscendInputXDim(dim_size, prim_name);
  if (x_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(dim_size, abstract::Shape::kShapeDimAny));
  }

  std::vector<int64_t> ori_paddings_val;
  auto padding_type = input_args[kInputIndex1]->GetType();
  if (padding_type->isa<TensorType>()) {
    auto paddings_shape_ptr = input_args[kInputIndex1]->GetShape();
    MS_EXCEPTION_IF_NULL(paddings_shape_ptr);
    if (paddings_shape_ptr->IsDynamic()) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>(dim_size, abstract::Shape::kShapeDimAny));
    }
    auto paddings_value = input_args[kInputIndex1]->GetValue();
    MS_EXCEPTION_IF_NULL(paddings_value);
    if (paddings_value->ContainsValueAny()) {
      return PaddingNoTensor(paddings_shape_ptr, x_shape);
    }
    ori_paddings_val =
      CheckAndConvertUtils::CheckTensorIntValue("paddings value", paddings_value, prim_name, padding_type);
  } else if (padding_type->isa<Tuple>() || padding_type->isa<List>()) {
    ori_paddings_val = CheckAndConvertUtils::CheckIntOrTupleInt("paddings value", input_args[1], prim_name);
  } else {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(dim_size, abstract::Shape::kShapeDimAny));
  }

  auto paddings_attr = PadV3DealWithPaddings(primitive, x_shape, ori_paddings_val);
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < dim_size; ++i) {
    auto index = dim_size - i - 1;
    int64_t now_dim_size = x_shape[i] + paddings_attr[index].first + paddings_attr[index].second;
    (void)CheckAndConvertUtils::CheckInteger("output size", now_dim_size, kGreaterThan, 0, prim_name);
    (void)out_shape.emplace_back(now_dim_size);
  }

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr PadV3InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  std::map<std::string, TypePtr> args = {{"x", input_args[0]->GetType()}};
  auto mode = GetValue<string>(prim->GetAttr("mode"));
  if (mode == kConstant) {
    return CheckAndConvertUtils::CheckTensorTypeSame(
      args,
      {kInt, kInt8, kInt16, kInt32, kInt64, kUInt, kUInt8, kUInt16, kFloat, kBFloat16, kFloat16, kFloat32, kFloat64,
       kComplex64, kComplex128, kBool},
      prim->name());
  } else {
    return CheckAndConvertUtils::CheckTensorTypeSame(args,
                                                     {kInt, kInt8, kInt16, kInt32, kInt64, kUInt, kUInt8, kUInt16,
                                                      kFloat, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128},
                                                     prim->name());
  }
}
}  // namespace

AbstractBasePtr PadV3Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kOtherInput = 2;
  auto mode = GetValue<string>(primitive->GetAttr("mode"));
  if (mode == kConstant) {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kConstantInput, primitive->name());
  } else {
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kOtherInput, primitive->name());
    if (input_args.size() == kConstantInput) {
      auto type_constant_value = input_args[kIndex2]->GetType();
      if (!type_constant_value->isa<TypeNone>()) {
        MS_EXCEPTION(ValueError)
          << "For '" << primitive->name()
          << "', the input[constant_value] is only valid when the attribute[mode] is `constant`. DO NOT set it in ["
          << mode << "] mode.";
      }
    }
  }
  auto infer_type = PadV3InferType(primitive, input_args);
  auto infer_shape = PadV3InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

bool PadV3::get_paddings_contiguous() const { return GetValue<bool>(GetAttr("paddings_contiguous")); }
std::string PadV3::get_mode() const { return GetValue<string>(GetAttr("mode")); }

MIND_API_OPERATOR_NAME_IMPL(PadV3, kNamePadV3, BaseOperator);

// AG means auto generated
class OPS_API AGPadV3Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PadV3Infer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {kInputIndex1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PadV3, prim::kPrimPadV3, AGPadV3Infer, false);
}  // namespace ops
}  // namespace mindspore
