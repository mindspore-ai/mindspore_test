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
int64_t GetOutputHWConvStr(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
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
    MS_EXCEPTION(ValueError) << "For primitive[ConvolutionStr], input padding string must be one of {'same', 'valid'}";
  }

  int64_t dim = SizeToLong(weight_shape.size()) - 2;
  std::vector<int64_t> padding = std::vector<int64_t>(dim, 0);
  return (input_shape[shape_pos] + 2 * padding[i] - dilation[i] * (weight_shape[shape_pos] - 1) - 1) / stride[i] + 1;
}

inline void IndicesCheckPositiveVectorConvStr(const string &arg_name, const ArrayValue<int64_t> &array,
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

BaseShapePtr ConvolutionStrFuncImpl::DynamicRankInfer(const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto weight_shape_ptr = input_args[kIndex1]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &weight_shape = weight_shape_ptr->GetShapeVector();
  auto padding_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());
  if (!IsDynamicRank(input_shape)) {
    int64_t input_dim = SizeToLong(input_shape.size());
    auto output_shape = ShapeVector(input_dim, abstract::Shape::kShapeDimAny);
    output_shape[0] = input_shape[kIndex0];
    if (padding_opt.has_value()) {
      mindspore::PadMode padding_enum_value = static_cast<mindspore::PadMode>(padding_opt.value());
      if (padding_enum_value == PadMode::SAME) {
        for (int i = 2; i < input_dim; i++) {
          output_shape[i] = input_shape[i];
        }
      }
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
  if (!IsDynamicRank(weight_shape)) {
    int64_t weight_dim = SizeToLong(weight_shape.size());
    auto output_shape = ShapeVector(weight_dim, abstract::Shape::kShapeDimAny);
    output_shape[1] = weight_shape[kIndex0];
    return std::make_shared<abstract::Shape>(output_shape);
  }
  std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
  return std::make_shared<abstract::Shape>(output_shape);
}

BaseShapePtr ConvolutionStrFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto weight_shape_ptr = input_args[kIndex1]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &weight_shape = weight_shape_ptr->GetShapeVector();
  auto padding_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());

  if (IsDynamicRank(input_shape) || IsDynamicRank(weight_shape)) {
    return DynamicRankInfer(input_args);
  }

  int64_t nd_output_shape_len = SizeToLong(weight_shape.size());
  auto nd_output_shape = ShapeVector(nd_output_shape_len, abstract::Shape::kShapeDimAny);
  nd_output_shape[0] = input_shape[kIndex0];
  nd_output_shape[1] = weight_shape[kIndex0];

  if (padding_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    auto transposed_opt = GetScalarValue<bool>(input_args[kIndex6]->BuildValue());
    if (transposed_opt.has_value()) {
      auto transposed = transposed_opt.value();
      if (transposed) {
        MS_EXCEPTION(ValueError) << "ConvolutionStr is not supported transposed=True";
      }
    }

    auto stride_opt = GetArrayValue<int64_t>(input_args[kIndex3]);
    auto dilation_opt = GetArrayValue<int64_t>(input_args[kIndex5]);
    if (!stride_opt.has_value() || !dilation_opt.has_value()) {
      if (padding_enum == PadMode::SAME) {
        for (int i = 2; i < nd_output_shape_len; i++) {
          nd_output_shape[i] = input_shape[i];
        }
      }
      return std::make_shared<abstract::Shape>(nd_output_shape);
    }
    const auto &stride = stride_opt.value();
    const auto &dilation = dilation_opt.value();
    IndicesCheckPositiveVectorConvStr("stride", stride, prim_name, true);
    IndicesCheckPositiveVectorConvStr("dilation", dilation, prim_name, true);

    auto output_padding_opt = GetArrayValue<int64_t>(input_args[kIndex7]);
    if (output_padding_opt.has_value()) {
      const auto &output_padding = output_padding_opt.value();
      IndicesCheckPositiveVectorConvStr("output_padding", output_padding, prim_name, false);
    }

    if (!IsDynamic(input_shape) && !IsDynamic(weight_shape)) {
      abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
      abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);

      auto group_opt = GetScalarValue<int64_t>(input_args[kIndex8]->GetValue());
      if (stride_opt.has_value()) {
        int64_t groups = group_opt.value();
        auto in_channels = input_shape[kIndex1];
        (void)CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1);
        (void)CheckAndConvertUtils::CheckInteger("out_channels", nd_output_shape[1], kGreaterEqual, groups);
        (void)CheckAndConvertUtils::CheckInteger("out_channels/groups", nd_output_shape[1] % groups, kEqual, 0);
        (void)CheckAndConvertUtils::CheckInteger("in_channels/groups", in_channels / groups, kEqual,
                                                 weight_shape[kIndex1]);
      }

      if (!input_args[kIndex2]->isa<abstract::AbstractNone>()) {
        auto bias_shape_vec = input_args[kIndex2]->GetShape()->GetShapeVector();
        if (!IsDynamicRank(bias_shape_vec) && !IsDynamic(bias_shape_vec)) {
          int64_t bias_rank = SizeToLong(bias_shape_vec.size());
          const auto bias_shape_size = 1;
          (void)CheckAndConvertUtils::CheckInteger("bias rank", bias_rank, kEqual, bias_shape_size, prim_name);
          (void)CheckAndConvertUtils::CheckInteger("bias of size", bias_shape_vec[kIndex0], kEqual,
                                                   weight_shape[kIndex0]);
        }
      }
    }

    for (int i = 2; i < nd_output_shape_len; i++) {
      nd_output_shape[i] = GetOutputHWConvStr(input_shape, weight_shape, i, i - 2, stride, padding_enum, dilation);
    }
    return std::make_shared<abstract::Shape>(nd_output_shape);
  } else {
    return std::make_shared<abstract::Shape>(nd_output_shape);
  }
}

TypePtr ConvolutionStrFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
