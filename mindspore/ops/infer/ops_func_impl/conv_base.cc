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

#include "infer/ops_func_impl/conv_base.h"
#include <optional>
#include <string>
#include <set>
#include <utility>
#include "abstract/dshape.h"
#include "include/common/utils/utils.h"
#include "mindapi/base/shape_vector.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kConvGap2 = 2;
constexpr auto kConvSize3 = 3;
constexpr auto kConvSize5 = 5;
}  // namespace
namespace conv_base {
int64_t ConvBaseGetOutputSpatialDim(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos,
                                    size_t i, const ArrayValue<int64_t> &stride, const ArrayValue<int64_t> &padding,
                                    const ArrayValue<int64_t> &dilation, bool transposed,
                                    const std::optional<ArrayValue<int64_t>> &output_padding_opt) {
  auto i_stride = i % stride.size();
  auto i_padding = i % padding.size();
  auto i_dilation = i % dilation.size();
  if (input_shape[shape_pos] == abstract::Shape::kShapeDimAny ||
      weight_shape[shape_pos] == abstract::Shape::kShapeDimAny || padding.IsValueUnknown(i_padding) ||
      dilation.IsValueUnknown(i_dilation) || stride.IsValueUnknown(i_stride)) {
    return abstract::Shape::kShapeDimAny;
  }

  if (!transposed) {
    return (input_shape[shape_pos] + 2 * padding[i_padding] - dilation[i_dilation] * (weight_shape[shape_pos] - 1) -
            1) /
             stride[i_stride] +
           1;
  } else {
    const auto &output_padding = output_padding_opt.value();
    auto i_output_padding = i % output_padding.size();
    if (output_padding.IsValueUnknown(i_output_padding)) {
      return abstract::Shape::kShapeDimAny;
    }
    return (input_shape[shape_pos] - 1) * stride[i_stride] - 2 * padding[i_padding] +
           dilation[i_dilation] * (weight_shape[shape_pos] - 1) + output_padding[i_output_padding] + 1;
  }
}

inline void ConvBaseIndicesCheckPositiveVector(const string &arg_name, const ArrayValue<int64_t> &array,
                                               const string &prim_name, bool exclude_zeros, size_t spatial_len) {
  if (MS_UNLIKELY(array.size() != kIndex1 && array.size() != spatial_len)) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", expected '" << arg_name
                             << "' to be a single integer value or a list of " << spatial_len << " values, but it's "
                             << array.ToString() << ".";
  }
  for (size_t i = 0; i < array.size(); ++i) {
    if (MS_UNLIKELY(array.IsValueUnknown(i))) {
      continue;
    }
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

void ConvBaseCheckRange(const std::string &arg_name, int64_t arg_value, int64_t up_bound, int64_t low_bound,
                        const std::string &prim_name) {
  if (arg_value < low_bound || arg_value > up_bound) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the " << arg_name << " must be equal to ["
                             << low_bound << ", " << up_bound << "], but got " << arg_value << ".";
  }
}
}  // namespace conv_base
void ConvBaseFunImpl::FetchSpatialDim(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                      const ShapeVector &input_shape, const ShapeVector &weight_shape, bool transposed,
                                      ShapeVector *const nd_output_shape) const {
  auto stride_opt = input_infos[idxes_.stride_idx]->GetArrayValue<int64_t>();
  auto padding_opt = input_infos[idxes_.padding_idx]->GetArrayValue<int64_t>();
  auto dilation_opt = input_infos[idxes_.dilation_idx]->GetArrayValue<int64_t>();
  std::optional<ArrayValue<int64_t>> output_padding_opt =
    transposed ? input_infos[idxes_.output_padding_idx]->GetArrayValue<int64_t>() : std::nullopt;
  if (MS_UNLIKELY(input_infos[idxes_.input_idx]->IsDynamicRank() || input_infos[idxes_.weight_idx]->IsDynamicRank() ||
                  !stride_opt.has_value() || !padding_opt.has_value() || !dilation_opt.has_value() ||
                  (transposed && !output_padding_opt.has_value()))) {
    MS_LOG(DEBUG) << "input dynamic rank: " << input_infos[idxes_.input_idx]->IsDynamicRank()
                  << ", weight dynamic rank: " << input_infos[idxes_.weight_idx]->IsDynamicRank()
                  << ", stride has_value:" << stride_opt.has_value()
                  << ", paddind has_value:" << padding_opt.has_value()
                  << ", dilation has_value:" << dilation_opt.has_value()
                  << ", output_padding has_value:" << output_padding_opt.has_value()
                  << ", output_shape:" << (*nd_output_shape);
    return;
  }

  const auto &prim_name = primitive->name();
  auto spatial_len = nd_output_shape->size() - kIndex2;

  const auto &stride = stride_opt.value();
  const auto &padding = padding_opt.value();
  const auto &dilation = dilation_opt.value();
  (void)CheckAndConvertUtils::CheckInteger("stride size", spatial_len, kGreaterEqual, SizeToLong(stride.size()),
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dilation size", spatial_len, kGreaterEqual, SizeToLong(dilation.size()),
                                           prim_name);
  conv_base::ConvBaseIndicesCheckPositiveVector("stride", stride, prim_name, true, spatial_len);
  conv_base::ConvBaseIndicesCheckPositiveVector("padding", padding, prim_name, false, spatial_len);
  conv_base::ConvBaseIndicesCheckPositiveVector("dilation", dilation, prim_name, true, spatial_len);
  if (transposed) {
    conv_base::ConvBaseIndicesCheckPositiveVector("output_padding", output_padding_opt.value(), prim_name, false,
                                                  spatial_len);
  }

  for (size_t i = 0; i < spatial_len; i++) {
    (*nd_output_shape)[i + kConvGap2] = conv_base::ConvBaseGetOutputSpatialDim(
      input_shape, weight_shape, i + kConvGap2, i, stride, padding, dilation, transposed, output_padding_opt);
  }
}

void ConvBaseFunImpl::FetchChannelDim(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                      const ShapeVector &input_shape, const ShapeVector &weight_shape, bool transposed,
                                      ShapeVector *const nd_output_shape) const {
  if (transposed) {
    auto &groups_info_ptr = input_infos[idxes_.groups_idx];
    auto groups_opt = groups_info_ptr->GetScalarValue<int64_t>();
    if (groups_opt.has_value() &&
        (!input_infos[idxes_.weight_idx]->IsDynamicRank() && weight_shape[kIndex1] != abstract::Shape::kShapeDimAny)) {
      auto groups = groups_opt.value();
      MS_CHECK_VALUE(groups >= 1,
                     CheckAndConvertUtils::FormatCheckIntegerMsg("groups", groups, kGreaterEqual, 1, primitive));
      (*nd_output_shape)[kIndex1] = weight_shape[kIndex1] * groups;
    }
  } else {
    (*nd_output_shape)[kIndex1] =
      !input_infos[idxes_.weight_idx]->IsDynamicRank() ? weight_shape.at(kIndex0) : abstract::Shape::kShapeDimAny;
  }
}

ShapeVector ConvBaseFunImpl::FetchBatchDim(const PrimitivePtr &primitive, const ShapeVector &input_shape,
                                           const ShapeVector &weight_shape, bool is_input_dyn_rank,
                                           bool is_weight_dyn_rank) const {
  if (MS_UNLIKELY(is_input_dyn_rank && is_weight_dyn_rank)) {
    return std::vector<int64_t>{abstract::Shape::kShapeRankAny};
  }
  if (MS_LIKELY(!(is_input_dyn_rank || is_weight_dyn_rank))) {
    MS_CHECK_VALUE(
      input_shape.size() == weight_shape.size(),
      CheckAndConvertUtils::FormatCheckIntegerMsg("rank of input and weight", SizeToLong(input_shape.size()), kEqual,
                                                  SizeToLong(weight_shape.size()), primitive));
  }

  std::vector<int64_t> nd_output_shape;
  if (MS_LIKELY(!is_weight_dyn_rank)) {
    conv_base::ConvBaseCheckRange("weight rank", SizeToLong(weight_shape.size()), kConvSize5, kConvSize3,
                                  "Convolution");
    nd_output_shape.resize(weight_shape.size(), abstract::Shape::kShapeDimAny);
  }
  if (MS_LIKELY(!is_input_dyn_rank)) {
    conv_base::ConvBaseCheckRange("input rank", SizeToLong(input_shape.size()), kConvSize5, kConvSize3, "Convolution");
    if (nd_output_shape.empty()) {
      nd_output_shape.resize(input_shape.size(), abstract::Shape::kShapeDimAny);
    }
    nd_output_shape[kIndex0] = input_shape.at(kIndex0);
  }
  return nd_output_shape;
}

std::pair<ShapeVector, bool> ConvBaseFunImpl::Batchify(const ShapeVector &input_shape, int64_t num_spatial_dims,
                                                       const std::string &prim_name) const {
  if (MS_UNLIKELY(IsDynamicRank(input_shape))) {
    MS_LOG(EXCEPTION) << "For ConvBaseFunImpl::Batchify, the input_shape should not be dynamic rank, but got "
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

ShapeVector ConvBaseFunImpl::ConvNdInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                              const ShapeVector &input_shape, const ShapeVector &weight_shape,
                                              std::optional<bool> transposed_opt) const {
  // check shapes of input and weight
  if (MS_LIKELY(!input_infos[idxes_.input_idx]->IsDynamicRank())) {
    abstract::CheckShapeAnyAndPositive(primitive->name() + " input_shape", input_shape);
  }
  if (MS_LIKELY(!input_infos[idxes_.weight_idx]->IsDynamicRank())) {
    abstract::CheckShapeAnyAndPositive(primitive->name() + " weight_shape", weight_shape);
  }
  // infer nd output shape
  auto nd_output_shape =
    FetchBatchDim(primitive, input_shape, weight_shape, input_infos[idxes_.input_idx]->IsDynamicRank(),
                  input_infos[idxes_.weight_idx]->IsDynamicRank());
  // 'Co/Ho/Wo' is unknown, if transposed is any value
  if (MS_UNLIKELY(IsDynamicRank(nd_output_shape) || !transposed_opt.has_value())) {
    return nd_output_shape;
  }
  auto transposed = transposed_opt.value();
  FetchChannelDim(primitive, input_infos, input_shape, weight_shape, transposed, &nd_output_shape);
  FetchSpatialDim(primitive, input_infos, input_shape, weight_shape, transposed, &nd_output_shape);
  return nd_output_shape;
}

ShapeArray ConvBaseFunImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input_shape = input_infos[idxes_.input_idx]->GetShape();
  const auto &weight_shape = input_infos[idxes_.weight_idx]->GetShape();
  auto transposed_opt = input_infos[idxes_.transposed_idx]->GetScalarValue<bool>();
  auto nd_output_shape = ConvNdInferShape(primitive, input_infos, input_shape, weight_shape, transposed_opt);
  return {std::move(nd_output_shape)};
}

std::vector<TypeId> ConvBaseFunImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
