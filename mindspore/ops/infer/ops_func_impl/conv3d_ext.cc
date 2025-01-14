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

#include "infer/ops_func_impl/conv3d_ext.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv3dInputArgsSize = 7;
constexpr size_t kConv3dStridePaddingDilationLen = 3;
constexpr int64_t kConv3dExtSize5 = 5;

int64_t GetOutputHWConv3d(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
                          const ArrayValue<int64_t> &stride, const ArrayValue<int64_t> &padding,
                          const ArrayValue<int64_t> &dilation, bool transposed, const ShapeVector &output_padding) {
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
    return (input_shape[shape_pos] - 1) * stride[i_stride] - 2 * padding[i_padding] +
           dilation[i_dilation] * (weight_shape[shape_pos] - 1) + output_padding[i] + 1;
  }
}

inline void IndicesCheckPositiveVectorConv3d(const string &arg_name, const ArrayValue<int64_t> &array,
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

void CheckRangeConv3d(const std::string &arg_name, int64_t arg_value, int64_t up_bound, int64_t low_bound,
                      const std::string &prim_name) {
  if (arg_value < low_bound || arg_value > up_bound) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the " << arg_name << " must be equal to ["
                             << low_bound << ", " << up_bound << "], but got " << arg_value << ".";
  }
}
}  // namespace

ShapeArray Conv3DExtFuncImpl::InferDynamicRank(const ShapeVector &input_shape, const ShapeVector &weight_shape) const {
  if (!IsDynamicRank(weight_shape)) {
    auto weight_shape_size = SizeToLong(weight_shape.size());
    CheckRangeConv3d("weight rank", weight_shape_size, kIndex5, kIndex3, "Convolution");
    auto output_shape = ShapeVector(weight_shape_size, abstract::Shape::kShapeDimAny);
    output_shape[1] = weight_shape[0];
    return {output_shape};
  }
  if (!IsDynamicRank(input_shape)) {
    auto output_shape = {input_shape[0], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                         abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    return {output_shape};
  }
  std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
  return {output_shape};
}

ShapeArray Conv3DExtFuncImpl::ConvNdInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                               const ShapeVector &input_shape, const ShapeVector &weight_shape) const {
  int64_t input_rank = SizeToLong(input_shape.size());
  int64_t weight_rank = SizeToLong(weight_shape.size());
  bool is_batch = input_rank == weight_rank ? true : false;
  ShapeVector output_shape(input_rank, abstract::Shape::kShapeDimAny);
  if (is_batch) {
    output_shape[0] = input_shape[0];
    return ConvNdCommonInferShape(primitive, input_infos, input_shape, weight_shape, output_shape);
  } else {
    auto _input_shape = input_shape;
    _input_shape.insert(_input_shape.begin(), 1);
    output_shape.insert(output_shape.begin(), 1);
    auto real_shape_vector_array =
      ConvNdCommonInferShape(primitive, input_infos, _input_shape, weight_shape, output_shape);
    if (SizeToLong(real_shape_vector_array.size()) < 1) {
      MS_LOG(EXCEPTION) << "For [Conv3DExt], the size of shape is less zero from function[ConvNdCommonInferShape]";
    }
    auto real_shape_vector = real_shape_vector_array[0];
    real_shape_vector.erase(real_shape_vector.begin());
    return {real_shape_vector};
  }
}

ShapeArray Conv3DExtFuncImpl::ConvNdCommonInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                                     const ShapeVector &input_shape, const ShapeVector &weight_shape,
                                                     const ShapeVector &output_shpe) const {
  auto prim_name = primitive->name();
  auto nd_output_shape = output_shpe;
  nd_output_shape[1] = weight_shape[0];

  auto stride_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
  auto padding_opt = input_infos[kIndex4]->GetArrayValue<int64_t>();
  auto dilation_opt = input_infos[kIndex5]->GetArrayValue<int64_t>();
  if (!stride_opt.has_value() || !padding_opt.has_value() || !dilation_opt.has_value()) {
    MS_LOG(DEBUG) << "For [Conv3DExt], stride has_value:" << stride_opt.has_value()
                  << ", paddind has_value:" << padding_opt.has_value()
                  << ", dilation has_value:" << dilation_opt.has_value() << ", output_shape:" << nd_output_shape;
    return {nd_output_shape};
  }

  const auto &stride = stride_opt.value();
  const auto &padding = padding_opt.value();
  const auto &dilation = dilation_opt.value();

  if (!IsDynamic(input_shape) && !IsDynamic(weight_shape)) {
    abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
    abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);
    auto weight_dim = SizeToLong(weight_shape.size());
    if (weight_dim != kConv3dExtSize5) {
      MS_LOG(EXCEPTION) << "The dim of argument[weight] must be equal 5, but got " << weight_dim;
    }
    IndicesCheckPositiveVectorConv3d("stride", stride, prim_name, true);
    IndicesCheckPositiveVectorConv3d("padding", padding, prim_name, false);
    IndicesCheckPositiveVectorConv3d("dilation", dilation, prim_name, true);

    auto group_opt = input_infos[kIndex6]->GetScalarValue<int64_t>();
    int64_t groups = group_opt.value();
    (void)CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1);
    auto out_channels = weight_shape[kIndex0];
    (void)CheckAndConvertUtils::CheckInteger("out_channels", out_channels, kGreaterEqual, groups);
    (void)CheckAndConvertUtils::CheckInteger("out_channels mod groups", out_channels % groups, kEqual, 0);

    std::vector<int64_t> input_shape_with_padding;
    std::vector<int64_t> kernel_shape_with_dilation;

    auto in_channels = input_shape[kIndex1];
    if (in_channels / groups != weight_shape[kIndex1]) {
      MS_EXCEPTION(ValueError) << "The argument error. in_channels/groups must be equal weight[1], "
                               << "but in_channels/groups is " << in_channels / groups << ", and weight[1] is "
                               << weight_shape[kIndex1];
    }
    int64_t input_rank = SizeToLong(input_shape.size());
    for (int64_t i = 2; i < input_rank; i++) {
      input_shape_with_padding.push_back(input_shape[i] + 2 * padding[(i - 2) % padding.size()]);
      kernel_shape_with_dilation.push_back(dilation[(i - 2) % dilation.size()] * (weight_shape[i] - 1) + 1);
      if (input_shape_with_padding.back() < kernel_shape_with_dilation.back()) {
        MS_EXCEPTION(ValueError) << "For [Conv3DExt], (Input_shape[i]{" << input_shape[i] << "} + 2 * padding[i-2]{"
                                 << padding[(i - 2) % padding.size()] << "})can't be less then "
                                 << "(delation[i-2]{" << dilation[(i - 2) % dilation.size()] << "} * (weight_shape[i]{"
                                 << weight_shape[i] << "} - 1) + 1).";
      }
    }
  }

  auto feature_len = SizeToLong(nd_output_shape.size()) - 2;
  for (int i = 0; i < feature_len; i++) {
    nd_output_shape[i + 2] =
      GetOutputHWConv3d(input_shape, weight_shape, i + 2, i, stride, padding, dilation, false, {0, 0, 0});
  }
  return {nd_output_shape};
}

ShapeArray Conv3DExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  if (input_infos.size() != kConv3dInputArgsSize) {
    MS_LOG(EXCEPTION) << "For [Conv3DExt], input args size should be  " << kConv3dInputArgsSize << ", but got "
                      << input_infos.size();
  }

  auto &input_tensor = input_infos[kIndex0];
  auto &weight_tensor = input_infos[kIndex1];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  if (IsDynamicRank(input_shape) || IsDynamicRank(weight_shape)) {
    return InferDynamicRank(input_shape, weight_shape);
  }

  int64_t input_rank = SizeToLong(input_shape.size());
  if (input_rank != kConv3dExtSize5 && input_rank != kConv3dExtSize5 - 1) {
    MS_EXCEPTION(ValueError) << "For [Conv3d], the dim of argument[input] must be equal to 4 or 5.";
  }

  int64_t weight_rank = SizeToLong(weight_shape.size());
  int64_t conv_dim = weight_rank - 2;
  if (conv_dim == 1 || conv_dim == 2 || conv_dim == 3) {
    return ConvNdInferShape(primitive, input_infos, input_shape, weight_shape);
  } else {
    MS_LOG(EXCEPTION) << "For [Conv3DExt], The weight size is must to equal 5 for Conv3d, but get " << weight_rank;
  }
}

std::vector<TypeId> Conv3DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
