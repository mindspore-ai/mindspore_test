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

#include "infer/ops_func_impl/conv3d_padding.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kConv3dPaddingGap2 = 2;
constexpr auto kConv3dPaddingSize3 = 3;
constexpr auto kConv3dPaddingSize5 = 5;

void CheckConvPaddingRestrictions(mindspore::PadMode mode, const ShapeVector &input_shape,
                                  const ShapeVector &weight_shape, const ArrayValue<int64_t> &stride,
                                  const ArrayValue<int64_t> &dilation) {
  if (mode == PadMode::VALID) {
    for (int i = 0; i < kConv3dPaddingSize3; i++) {
      if (!dilation.IsValueUnknown(i % dilation.size())) {
        if (input_shape[i + kConv3dPaddingGap2] -
              (weight_shape[i + kConv3dPaddingGap2] - 1) * dilation[i % dilation.size()] - 1 <
            0) {
          MS_EXCEPTION(ValueError)
            << "For primitive[Conv3DPadding], (Hin + PadUp + PadDown - (kh - 1) * DilationH - 1), "
            << "(Win + PadLeft + PadRight - (kw - 1) * DilationW - 1) and (Din + PadFront + PadBack - (kd - 1) * "
               "DilationD - 1)"
            << " must greater than 0.";
        }
      }
    }
    return;
  }
  for (int i = 0; i < kConv3dPaddingSize3; i++) {
    if (dilation.IsValueUnknown(i % dilation.size()) || stride.IsValueUnknown(i % stride.size())) {
      continue;
    }
    auto pads = (input_shape[i + kConv3dPaddingGap2] - 1) * stride[i % stride.size()] +
                (weight_shape[i + kConv3dPaddingGap2] - 1) * dilation[i % dilation.size()] + 1 -
                input_shape[i + kConv3dPaddingGap2];
    if (pads / 2 >= weight_shape[i + kConv3dPaddingGap2]) {
      MS_EXCEPTION(ValueError) << "For primitive[Conv3DPadding], pad should be less "
                               << weight_shape[i + kConv3dPaddingGap2] << ", but got " << pads / 2
                               << ". Taking the H dimension as an example, when pad is filled symmetrically,"
                               << " the calculation of the pad value can be obtained by "
                               << "((Hout - 1) * strideH + (Hk - 1) * DilationH + 1 - Hin) / 2";
    }
  }
}

int64_t GetOutputHWPadding(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
                           const ArrayValue<int64_t> &stride, const mindspore::PadMode &padding_enum,
                           const ArrayValue<int64_t> &dilation) {
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
    MS_EXCEPTION(ValueError) << "For primitive[Conv3DPadding], input padding string must be one of {'same', 'valid'}";
  }
  int64_t feature_len = SizeToLong(input_shape.size()) - 2;
  std::vector<int64_t> padding(feature_len, 0);
  return (input_shape[shape_pos] + 2 * padding[i] - dilation[i_dilation] * (weight_shape[shape_pos] - 1) - 1) /
           stride[i_stride] +
         1;
}

inline void IndicesCheckPositiveVectorPadding(const string &arg_name, const ArrayValue<int64_t> &array,
                                              const string &prim_name, bool exclude_zeros) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array.IsValueUnknown(i)) {
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

void CheckRangePadding(const std::string &arg_name, int64_t arg_value, int64_t up_bound, int64_t low_bound,
                       const std::string &prim_name) {
  if (arg_value < low_bound || arg_value > up_bound) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the " << arg_name << " must be equal to ["
                             << low_bound << ", " << up_bound << "], but got " << arg_value << ".";
  }
}
}  // namespace

ShapeArray Conv3DPaddingFuncImpl::InferDynamicRank(const ShapeVector &input_shape,
                                                   const ShapeVector &weight_shape) const {
  if (!IsDynamicRank(weight_shape)) {
    auto weight_shape_size = SizeToLong(weight_shape.size());
    CheckRangePadding("weight rank", weight_shape_size, kConv3dPaddingSize5, kConv3dPaddingSize3, "Conv3dPadding");
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

ShapeArray Conv3DPaddingFuncImpl::ConvNdCommonInferShape(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos,
                                                         const ShapeVector &input_shape,
                                                         const ShapeVector &weight_shape,
                                                         const ShapeVector &output_shpe) const {
  auto prim_name = primitive->name();
  auto nd_output_shape = output_shpe;
  int64_t nd_output_shape_len = SizeToLong(nd_output_shape.size());
  nd_output_shape[0] = input_shape[kIndex0];
  nd_output_shape[1] = weight_shape[kIndex0];

  auto padding_opt = input_infos[kIndex4]->GetScalarValue<int64_t>();
  if (padding_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    auto stride_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
    auto dilation_opt = input_infos[kIndex5]->GetArrayValue<int64_t>();
    if (!stride_opt.has_value() || !dilation_opt.has_value()) {
      if (padding_enum == PadMode::SAME) {
        for (int i = 2; i < nd_output_shape_len; i++) {
          nd_output_shape[i] = input_shape[i];
        }
      }
      MS_LOG(DEBUG) << "For primitive[Conv3DPadding], stride has_value:" << stride_opt.has_value()
                    << ", dilation has_value:" << dilation_opt.has_value() << ", output_shape:" << nd_output_shape;
      return {nd_output_shape};
    }
    const auto &stride = stride_opt.value();
    const auto &dilation = dilation_opt.value();
    if (padding_enum == PadMode::SAME) {
      for (int i = 0; i < SizeToLong(stride.size()); i++) {
        if (!stride.IsValueUnknown(i)) {
          if (stride[i] != 1) {
            MS_EXCEPTION(ValueError) << "For primitive[Conv3DPadding], stride must be 1 when padding is same.";
          }
        }
      }
    }
    IndicesCheckPositiveVectorPadding("stride", stride, prim_name, true);
    IndicesCheckPositiveVectorPadding("dilation", dilation, prim_name, true);

    if (!IsDynamic(input_shape) && !IsDynamic(weight_shape)) {
      auto weight_dim = SizeToLong(weight_shape.size());
      if (weight_dim != kConv3dPaddingSize5) {
        MS_LOG(EXCEPTION) << "The dim of argument[weight] must be equal 5, but got " << weight_dim;
      }
      (void)CheckAndConvertUtils::CheckInteger("The dim of weight", weight_shape.size(), kGreaterEqual,
                                               kConv3dPaddingSize5);
      abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
      abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);

      auto group_opt = input_infos[kIndex6]->GetScalarValue<int64_t>();
      if (stride_opt.has_value()) {
        int64_t groups = group_opt.value();
        auto in_channels = input_shape[kIndex1];
        (void)CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1);
        (void)CheckAndConvertUtils::CheckInteger("out_channels", nd_output_shape[1], kGreaterEqual, groups);
        (void)CheckAndConvertUtils::CheckInteger("out_channels mod groups", nd_output_shape[1] % groups, kEqual, 0);
        if (in_channels / groups != weight_shape[kIndex1]) {
          MS_EXCEPTION(ValueError) << "The argument error. in_channels/groups must be equal weight[1], "
                                   << "but in_channels/groups is " << in_channels / groups << ", and weight[1] is "
                                   << weight_shape[kIndex1];
        }
      }
      CheckConvPaddingRestrictions(padding_enum, input_shape, weight_shape, stride, dilation);
    }

    for (int i = 2; i < nd_output_shape_len; i++) {
      nd_output_shape[i] = GetOutputHWPadding(input_shape, weight_shape, i, i - 2, stride, padding_enum, dilation);
    }
    return {nd_output_shape};
  } else {
    return {nd_output_shape};
  }
}

ShapeArray Conv3DPaddingFuncImpl::ConvNdInferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                                   const ShapeVector &input_shape,
                                                   const ShapeVector &weight_shape) const {
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
      MS_LOG(EXCEPTION) << "For [Conv3DPadding], Infer shape array size is less zero from ConvNdCommonInferShape";
    }
    auto real_shape_vector = real_shape_vector_array[0];
    real_shape_vector.erase(real_shape_vector.begin());
    return {real_shape_vector};
  }
}

ShapeArray Conv3DPaddingFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_shape = input_infos[kIndex0]->GetShape();
  const auto weight_shape = input_infos[kIndex1]->GetShape();
  if (IsDynamicRank(input_shape) || IsDynamicRank(weight_shape)) {
    return InferDynamicRank(input_shape, weight_shape);
  }

  int64_t input_rank = SizeToLong(input_shape.size());
  if (input_rank != kConv3dPaddingSize5 && input_rank != kConv3dPaddingSize5 - 1) {
    MS_EXCEPTION(ValueError) << "For [Conv3d], the dim of argument[input] must be equal to 4 or 5.";
  }

  int64_t weight_rank = SizeToLong(weight_shape.size());
  int64_t conv_dim = weight_rank - 2;
  if (conv_dim == 1 || conv_dim == 2 || conv_dim == 3) {
    return ConvNdInferShape(primitive, input_infos, input_shape, weight_shape);
  } else {
    MS_LOG(EXCEPTION) << "For primitive[Conv3DPadding], The weight size is 2, 3, or 4.";
  }
}

std::vector<TypeId> Conv3DPaddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
