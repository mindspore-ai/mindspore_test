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

#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
void CheckConvStrRestrictions(mindspore::PadMode mode, const ShapeVector &input_shape, const ShapeVector &weight_shape,
                              const ArrayValue<int64_t> &stride, const ArrayValue<int64_t> &dilation) {
  if (mode == PadMode::VALID) {
    for (int i = 0; i < 3; i++) {
      if (!dilation.IsValueUnknown(i % dilation.size())) {
        if (input_shape[i + 2] - (weight_shape[i + 2] - 1) * dilation[i % dilation.size()] - 1 < 0) {
          MS_EXCEPTION(ValueError)
            << "For primitive[Conv3DStr], (Hin + PadUp + PadDown - (Hk - 1) * DilationH - 1), "
            << "(Win + PadLeft + PadRight - (Wk - 1) * DilationW - 1) and (Din + PadFront + PadBack - (Dk - 1) * "
               "DilationD - 1)"
            << " must greater than 0.";
        }
      }
    }
    return;
  }
  for (int i = 0; i < 3; i++) {
    if (dilation.IsValueUnknown(i % dilation.size()) || stride.IsValueUnknown(i % stride.size())) {
      continue;
    }
    auto pads = (input_shape[i + 2] - 1) * stride[i % stride.size()] +
                (weight_shape[i + 2] - 1) * dilation[i % dilation.size()] + 1 - input_shape[i + 2];
    if (pads / 2 >= weight_shape[i + 2]) {
      MS_EXCEPTION(ValueError) << "For primitive[Conv3DStr], pad should be less " << weight_shape[i + 2] << ", but got "
                               << pads / 2
                               << ". Taking the H dimension as an example, when pad is filled symmetrically,"
                               << " the calculation of the pad value can be obtained by "
                               << "((Hout - 1) * strideH + (Hk - 1) * DilationH + 1 - Hin) / 2";
    }
  }
}

int64_t GetOutputHWConvStr(const ShapeVector &input_shape, const ShapeVector &weight_shape, size_t shape_pos, size_t i,
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
    MS_EXCEPTION(ValueError) << "For primitive[ConvolutionStr], input padding string must be one of {'same', 'valid'}";
  }

  int64_t dim = SizeToLong(weight_shape.size()) - 2;
  std::vector<int64_t> padding = std::vector<int64_t>(dim, 0);
  return (input_shape[shape_pos] + 2 * padding[i] - dilation[i_dilation] * (weight_shape[shape_pos] - 1) - 1) /
           stride[i_stride] +
         1;
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

ShapeArray ConvolutionStrFuncImpl::DynamicRankInfer(const InferInfoPtrList &input_infos) const {
  auto &input_tensor = input_infos[kIndex0];
  auto &weight_tensor = input_infos[kIndex1];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  auto padding_opt = input_infos[kIndex4]->GetScalarValue<int64_t>();
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

ShapeArray ConvolutionStrFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  auto &input_tensor = input_infos[kIndex0];
  auto &weight_tensor = input_infos[kIndex1];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  auto padding_opt = input_infos[kIndex4]->GetScalarValue<int64_t>();

  if (input_tensor->IsDynamicRank() || weight_tensor->IsDynamicRank()) {
    return DynamicRankInfer(input_infos);
  }

  int64_t nd_output_shape_len = SizeToLong(weight_shape.size());
  auto nd_output_shape = ShapeVector(nd_output_shape_len, abstract::Shape::kShapeDimAny);
  nd_output_shape[kIndex0] = input_shape[kIndex0];
  nd_output_shape[kIndex1] = weight_shape[kIndex0];

  if (padding_opt.has_value()) {
    mindspore::PadMode padding_enum = static_cast<mindspore::PadMode>(padding_opt.value());
    auto transposed_opt = input_infos[kIndex6]->GetScalarValue<bool>();
    if (transposed_opt.has_value()) {
      auto transposed = transposed_opt.value();
      if (transposed) {
        MS_EXCEPTION(ValueError) << "ConvolutionStr is not supported transposed=True";
      }
    }

    auto stride_opt = input_infos[kIndex3]->GetArrayValue<int64_t>();
    auto dilation_opt = input_infos[kIndex5]->GetArrayValue<int64_t>();
    if (!stride_opt.has_value() || !dilation_opt.has_value()) {
      if (padding_enum == PadMode::SAME) {
        for (int i = 2; i < nd_output_shape_len; i++) {
          nd_output_shape[i] = input_shape[i];
        }
      }
      return {nd_output_shape};
    }
    const auto &stride = stride_opt.value();
    const auto &dilation = dilation_opt.value();
    IndicesCheckPositiveVectorConvStr("stride", stride, prim_name, true);
    IndicesCheckPositiveVectorConvStr("dilation", dilation, prim_name, true);

    auto output_padding_opt = input_infos[kIndex7]->GetArrayValue<int64_t>();
    if (output_padding_opt.has_value()) {
      const auto &output_padding = output_padding_opt.value();
      IndicesCheckPositiveVectorConvStr("output_padding", output_padding, prim_name, false);
    }

    if (!input_tensor->IsDynamic() && !weight_tensor->IsDynamic()) {
      abstract::CheckShapeAnyAndPositive(prim_name + " x_shape", input_shape);
      abstract::CheckShapeAnyAndPositive(prim_name + " w_shape", weight_shape);

      auto group_opt = input_infos[kIndex8]->GetScalarValue<int64_t>();
      if (stride_opt.has_value()) {
        int64_t groups = group_opt.value();
        auto in_channels = input_shape[kIndex1];
        (void)CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterEqual, 1);
        (void)CheckAndConvertUtils::CheckInteger("out_channels", nd_output_shape[kIndex1], kGreaterEqual, groups);
        (void)CheckAndConvertUtils::CheckInteger("out_channels/groups", nd_output_shape[kIndex1] % groups, kEqual, 0);
        (void)CheckAndConvertUtils::CheckInteger("in_channels/groups", in_channels / groups, kEqual,
                                                 weight_shape[kIndex1]);
      }

      if (!input_infos[kIndex2]->IsNone()) {
        auto bias_shape_vec = input_infos[kIndex2]->GetShape();
        if (input_infos[kIndex2]->IsDynamicRank() && input_infos[kIndex2]->IsDynamic()) {
          int64_t bias_rank = SizeToLong(bias_shape_vec.size());
          const auto bias_shape_size = 1;
          (void)CheckAndConvertUtils::CheckInteger("bias rank", bias_rank, kEqual, bias_shape_size, prim_name);
          (void)CheckAndConvertUtils::CheckInteger("bias of size", bias_shape_vec[kIndex0], kEqual,
                                                   weight_shape[kIndex0]);
        }
      }

      CheckConvStrRestrictions(padding_enum, input_shape, weight_shape, stride, dilation);
    }

    for (int i = 2; i < nd_output_shape_len; i++) {
      nd_output_shape[i] = GetOutputHWConvStr(input_shape, weight_shape, i, i - 2, stride, padding_enum, dilation);
    }
    return {nd_output_shape};
  } else {
    return {nd_output_shape};
  }
}

std::vector<TypeId> ConvolutionStrFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
