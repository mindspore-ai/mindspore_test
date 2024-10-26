/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/convolution_str.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int64_t GetOutputHW(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
                    const ArrayValue<int64_t> &stride, const mindspore::PadMode &padding_enum,
                    const ArrayValue<int64_t> &dilation) {
  if (input_shape[shape_pos] == abstract::Shape::kShapeDimAny ||
      weight_shape[shape_pos] == abstract::Shape::kShapeDimAny || dilation.IsValueUnknown(i) ||
      stride.IsValueUnknown(i)) {
    return abstract::Shape::kShapeDimAny;
  }
  if (padding_enum == PadMode::SAME) {
    return input_shape[shape_pos];
  } else if (padding_enum != PadMode::VALID) {
    MS_EXCEPTION(ValueError) << "Input padding string must be one of {'same', 'valid'}";
  }
  std::vector<int64_t> padding = {0, 0};
  return (input_shape[shape_pos] + 2 * padding[i] - dilation[i] * (weight_shape[shape_pos] - 1) - 1) / stride[i] + 1;
}

inline void IndicesCheckPositiveVector(const string &arg_name, const ArrayValue<int64_t> &array,
                                       const string &prim_name, bool exclude_zeros) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (exclude_zeros) {
      if (MS_UNLIKELY(array[i] <= 0)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", '" << arg_name << "' must be positive, but it's "
                                 << array.ToString() << ".";
      }
    } else {
      if (MS_UNLIKELY(array[i] < 0)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", '" << arg_name << "' must be not negetive, but it's "
                                 << array.ToString() << ".";
      }
    }
  }
}
}  // namespace

BaseShapePtr ConvolutionStrFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  const auto conv2d_shape_size = 4;
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto weight_shape_ptr = input_args[kIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  MS_EXCEPTION_IF_NULL(weight_shape_ptr);
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &weight_shape = weight_shape_ptr->GetShapeVector();
  auto padding_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());
  int64_t Co = abstract::Shape::kShapeDimAny;
  int64_t Ho = abstract::Shape::kShapeDimAny;
  int64_t Wo = abstract::Shape::kShapeDimAny;
  int64_t N = abstract::Shape::kShapeDimAny;

  if (IsDynamicRank(input_shape) || IsDynamicRank(weight_shape)) {
    if (!IsDynamicRank(input_shape)) {
      (void)CheckAndConvertUtils::CheckInteger("input rank", SizeToLong(input_shape.size()), kEqual, conv2d_shape_size,
                                               prim_name);
      N = input_shape[kIndex0];
      if (padding_opt.has_value()) {
        mindspore::PadMode padding_enum_value = static_cast<mindspore::PadMode>(padding_opt.value());
        if (padding_enum_value == PadMode::SAME) {
          Ho = input_shape[kIndex2];
          Wo = input_shape[kIndex3];
        }
      }
      auto output_shape = {N, Co, Ho, Wo};
      return std::make_shared<abstract::Shape>(output_shape);
    }
    if (!IsDynamicRank(weight_shape)) {
      (void)CheckAndConvertUtils::CheckInteger("weight rank", SizeToLong(weight_shape.size()), kEqual,
                                               conv2d_shape_size, prim_name);
      Co = weight_shape[kIndex0];
      auto output_shape = {N, Co, Ho, Wo};
      return std::make_shared<abstract::Shape>(output_shape);
    }
    std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }

  // Support conv2d first
  int64_t input_rank = SizeToLong(input_shape.size());
  int64_t weight_rank = SizeToLong(weight_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("input rank", input_rank, kEqual, conv2d_shape_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("weight rank", weight_rank, kEqual, conv2d_shape_size, prim_name);

  N = input_shape[kIndex0];
  Co = weight_shape[kIndex0];

  if (padding_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    auto transposed_opt = GetScalarValue<bool>(input_args[kIndex6]->BuildValue());
    if (transposed_opt.has_value()) {
      auto transposed = transposed_opt.value();
      if (transposed) {
        MS_EXCEPTION(ValueError) << "ConvolutionStr is not supported transposed=Ture";
      }
    }

    auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex3]);
    auto dilation_opt = GetArrayValue<int64_t>(input_args[kIndex5]);
    if (!stride_opt.has_value() || !dilation_opt.has_value()) {
      if (padding_enum == PadMode::SAME) {
        Ho = input_shape[kIndex2];
        Wo = input_shape[kIndex3];
      }
      auto output_shape = {N, Co, Ho, Wo};
      MS_LOG(DEBUG) << "stride has_value:" << stride_opt.has_value()
                    << ", dilation has_value:" << dilation_opt.has_value() << ", output_shape:" << output_shape;
      return std::make_shared<abstract::Shape>(output_shape);
    }
    const auto &stride = stride_opt.value();
    const auto &dilation = dilation_opt.value();
    IndicesCheckPositiveVector("stride", stride, prim_name, true);
    IndicesCheckPositiveVector("dilation", dilation, prim_name, true);

    constexpr size_t h_begin_pos = 2;  // 'NCHW', the pos of 'H' is 2
    constexpr size_t w_begin_pos = 3;  // 'NCHW', the pos of 'W' is 3
    Ho = GetOutputHW(input_shape, weight_shape, h_begin_pos, 0, stride, padding_enum, dilation);
    Wo = GetOutputHW(input_shape, weight_shape, w_begin_pos, 1, stride, padding_enum, dilation);
    auto output_shape = {N, Co, Ho, Wo};
    return std::make_shared<abstract::Shape>(output_shape);
  } else {
    auto output_shape = {N, Co, Ho, Wo};
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr ConvolutionStrFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
