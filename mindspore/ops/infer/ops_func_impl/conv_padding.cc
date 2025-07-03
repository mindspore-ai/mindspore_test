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

#include "infer/ops_func_impl/conv_padding.h"

#include <utility>
#include <string>
#include <set>

#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kConvPaddingGap2 = 2;

std::pair<ShapeVector, bool> ConvPaddingFuncImpl::Batchify(const ShapeVector &input_shape, int64_t num_spatial_dims,
                                                           const std::string &prim_name) const {
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    MS_LOG(EXCEPTION) << "For ConvPaddingFuncImpl::Batchify, the input_shape should not be dynamic rank, but got "
                      << input_shape;
  }
  const auto dim_count_no_batch = num_spatial_dims + 1;
  const auto dim_count_batch = dim_count_no_batch + 1;
  auto origin_shape_dim = SizeToLong(input_shape.size());
  const auto is_batched = (origin_shape_dim == dim_count_batch);
  if (MS_UNLIKELY(origin_shape_dim != dim_count_no_batch && !is_batched)) {
    MS_LOG(EXCEPTION) << "Expected " << dim_count_no_batch << "D (unbatched) or " << dim_count_batch
                      << "D (batched) input to " << prim_name << ", but got input of shape: " << input_shape;
  }
  ShapeVector batched_input_shape(input_shape);
  if (!is_batched) {
    batched_input_shape.insert(batched_input_shape.begin(), 1, 1);
  }
  return std::make_pair(std::move(batched_input_shape), is_batched);
}

ShapeArray ConvPaddingFuncImpl::ConvNdInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                                 const ShapeVector &input_shape,
                                                 const ShapeVector &weight_shape) const {
  int64_t input_rank = SizeToLong(input_shape.size());
  int64_t weight_rank = SizeToLong(weight_shape.size());
  bool is_batch = input_rank == weight_rank ? true : false;
  ShapeVector output_shape(input_rank, abstract::Shape::kShapeDimAny);
  if (is_batch) {
    output_shape[kIndex0] = input_shape[kIndex0];
    return ConvNdPaddingCommonInferShape(primitive, input_infos, input_shape, weight_shape, output_shape);
  } else {
    auto _input_shape = input_shape;
    _input_shape.insert(_input_shape.begin(), 1);
    output_shape.insert(output_shape.begin(), 1);
    ShapeArray real_shape_vector_array{};
    real_shape_vector_array =
      ConvNdPaddingCommonInferShape(primitive, input_infos, _input_shape, weight_shape, output_shape);
    if (SizeToLong(real_shape_vector_array.size()) < 1) {
      MS_LOG(EXCEPTION) << "Infer shape array size is less zero from ConvNdCommonInferShape";
    }
    auto real_shape_vector = real_shape_vector_array[kIndex0];
    real_shape_vector.erase(real_shape_vector.begin());
    return {real_shape_vector};
  }
}

ShapeArray ConvPaddingFuncImpl::ConvNdPaddingCommonInferShape(const PrimitivePtr &primitive,
                                                              const InferInfoPtrList &input_infos,
                                                              const ShapeVector &input_shape,
                                                              const ShapeVector &weight_shape,
                                                              const ShapeVector &output_shpe) const {
  auto prim_name = primitive->name();
  auto nd_output_shape = output_shpe;
  nd_output_shape[kIndex0] = input_shape[kIndex0];
  nd_output_shape[kIndex1] = weight_shape[kIndex0];

  auto padding_opt = input_infos[idxes_.padding_idx]->GetScalarValue<int64_t>();
  if (padding_opt.has_value()) {
    auto feature_len = SizeToLong(nd_output_shape.size()) - 2;
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    auto stride_opt = input_infos[idxes_.stride_idx]->GetArrayValue<int64_t>();
    auto dilation_opt = input_infos[idxes_.dilation_idx]->GetArrayValue<int64_t>();
    if (!stride_opt.has_value() || !dilation_opt.has_value()) {
      if (padding_enum == PadMode::SAME) {
        for (int i = 0; i < feature_len; i++) {
          nd_output_shape[i + kConvPaddingGap2] = input_shape[i + kConvPaddingGap2];
        }
      }
      return {nd_output_shape};
    }
    const auto &stride = stride_opt.value();
    const auto &dilation = dilation_opt.value();
    MS_CHECK_VALUE(feature_len >= SizeToLong(stride.size()),
                   CheckAndConvertUtils::CheckInteger("stride size", feature_len, kGreaterEqual,
                                                      SizeToLong(stride.size()), prim_name));
    MS_CHECK_VALUE(feature_len >= SizeToLong(dilation.size()),
                   CheckAndConvertUtils::CheckInteger("dilation size", feature_len, kGreaterEqual,
                                                      SizeToLong(dilation.size()), prim_name));
    if (padding_enum == PadMode::SAME) {
      IndicesCheckPositiveVec("stride", stride, prim_name, true, true);
    } else {
      IndicesCheckPositiveVec("stride", stride, prim_name, true, false);
    }
    IndicesCheckPositiveVec("dilation", dilation, prim_name, true, false);

    if (!input_infos[idxes_.input_idx]->IsDynamic() && !input_infos[idxes_.weight_idx]->IsDynamic()) {
      abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
      abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);

      auto group_opt = input_infos[idxes_.groups_idx]->GetScalarValue<int64_t>();
      if (group_opt.has_value()) {
        int64_t groups = group_opt.value();
        auto in_channels = input_shape[kIndex1];
        auto out_channels = weight_shape[kIndex0];
        MS_CHECK_VALUE(groups >= 1, CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1));
        MS_CHECK_VALUE(out_channels >= groups,
                       CheckAndConvertUtils::CheckInteger("out_channels", out_channels, kGreaterEqual, groups));
        MS_CHECK_VALUE(out_channels % groups == 0,
                       CheckAndConvertUtils::CheckInteger("out_channels//groups", out_channels % groups, kEqual, 0));
        if (in_channels / groups != weight_shape[kIndex1]) {
          MS_EXCEPTION(ValueError) << "The argument error. in_channels/groups must be equal weight[1], "
                                   << "but in_channels/groups is " << in_channels / groups << ", and weight[1] is "
                                   << weight_shape[kIndex1];
        }
      }
    }
    for (int i = 0; i < feature_len; i++) {
      nd_output_shape[i + kConvPaddingGap2] =
        GetOutputHWPadding(input_shape, weight_shape, i + kConvPaddingGap2, i, stride, padding_enum, dilation);
    }
  }
  return {nd_output_shape};
}

void ConvPaddingFuncImpl::IndicesCheckPositiveVec(const string &arg_name, const ArrayValue<int64_t> &array,
                                                  const string &prim_name, bool exclude_zeros,
                                                  bool padding_stride) const {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array.IsValueUnknown(i)) {
      continue;
    }
    if (padding_stride) {
      if (MS_UNLIKELY(array[i] != 1)) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", '" << arg_name
                                 << "' must be 1, when padding is same, but it's " << array.ToString() << ".";
      }
    } else {
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
}

int64_t ConvPaddingFuncImpl::GetOutputHWPadding(const ShapeVector &input_shape, const ShapeVector &weight_shape,
                                                size_t shape_pos, size_t i, const ArrayValue<int64_t> &stride,
                                                const mindspore::PadMode &padding_enum,
                                                const ArrayValue<int64_t> &dilation) const {
  auto i_stride = i % stride.size();
  auto i_dilation = i % dilation.size();
  if (input_shape[shape_pos] == abstract::Shape::kShapeDimAny ||
      weight_shape[shape_pos] == abstract::Shape::kShapeDimAny || dilation.IsValueUnknown(i_dilation) ||
      stride.IsValueUnknown(i_stride)) {
    return abstract::Shape::kShapeDimAny;
  }
  if (padding_enum == PadMode::SAME) {
    return input_shape[shape_pos];
  } else if (padding_enum != PadMode::VALID) {
    MS_EXCEPTION(ValueError) << "Input padding string must be one of {'same', 'valid'}";
  }
  return (input_shape[shape_pos] - dilation[i_dilation] * (weight_shape[shape_pos] - 1) - 1) / stride[i_stride] + 1;
}

ShapeArray ConvPaddingFuncImpl::DynamicRankInfer(const InferInfoPtrList &input_infos) const {
  auto &input_tensor = input_infos[idxes_.input_idx];
  auto &weight_tensor = input_infos[idxes_.weight_idx];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  auto padding_opt = input_infos[idxes_.padding_idx]->GetScalarValue<int64_t>();
  if (MS_LIKELY(!input_tensor->IsDynamicRank())) {
    int64_t input_dim = SizeToLong(input_shape.size());
    auto output_shape = ShapeVector(input_dim, abstract::Shape::kShapeDimAny);
    output_shape[kIndex0] = input_shape[kIndex0];
    if (padding_opt.has_value()) {
      mindspore::PadMode padding_enum_value = static_cast<mindspore::PadMode>(padding_opt.value());
      if (padding_enum_value == PadMode::SAME) {
        for (int i = 2; i < input_dim; i++) {
          output_shape[i] = input_shape[i];
        }
      }
    }
    return {output_shape};
  }
  if (MS_LIKELY(!weight_tensor->IsDynamicRank())) {
    int64_t weight_dim = SizeToLong(weight_shape.size());
    auto output_shape = ShapeVector(weight_dim, abstract::Shape::kShapeDimAny);
    output_shape[kIndex1] = weight_shape[kIndex0];
    return {output_shape};
  }
  std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
  return {output_shape};
}

ShapeArray ConvPaddingFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input_tensor = input_infos[idxes_.input_idx];
  auto &weight_tensor = input_infos[idxes_.weight_idx];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  if (input_tensor->IsDynamicRank() || weight_tensor->IsDynamicRank()) {
    return DynamicRankInfer(input_infos);
  }
  return ConvNdInferShape(primitive, input_infos, input_shape, weight_shape);
}

std::vector<TypeId> ConvPaddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
