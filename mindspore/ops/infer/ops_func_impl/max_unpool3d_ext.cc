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

#include "infer/ops_func_impl/max_unpool3d_ext.h"
#include <set>
#include <memory>
#include <string>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {

static inline bool CheckMaxUnpool3DInputIsDynamic(const InferInfoPtrList &input_infos) {
  return input_infos[kInputIndex0]->IsDynamic() && input_infos[kInputIndex1]->IsDynamic();
}

static inline bool IsEqualToTargets(int64_t num, int target1, int target2) { return num == target1 || num == target2; }

static inline ShapeVector SetUnknownOutputShape(const ShapeVector &out_shape, size_t last_dim) {
  ShapeVector unknown_output_shape = out_shape;
  unknown_output_shape[last_dim - kDim2] = -1;
  unknown_output_shape[last_dim - kDim1] = -1;
  unknown_output_shape[last_dim] = -1;
  return unknown_output_shape;
}

static inline bool IsInRange(int64_t value, int64_t min, int64_t max) { return value >= min && value < max; }

static inline ShapeArray Cal3DOutShape(const InferInfoPtr &ksize_ptr, const InferInfoPtr &strides_ptr,
                                       const InferInfoPtr &pads_ptr, ShapeVector out_shape, size_t last_dim,
                                       const PrimitivePtr &primitive) {
  auto ksize_opt = ksize_ptr->GetArrayValue<int64_t>();
  auto strides_opt = (strides_ptr->IsNone()) ? ksize_opt : strides_ptr->GetArrayValue<int64_t>();
  auto pads_opt = pads_ptr->GetArrayValue<int64_t>();

  if (!(ksize_opt.has_value() && strides_opt.has_value() && pads_opt.has_value()) ||
      ksize_opt.value().HasUnknownValue() || strides_opt.value().HasUnknownValue() ||
      pads_opt.value().HasUnknownValue()) {
    return {SetUnknownOutputShape(out_shape, last_dim), {}};
  }

  std::vector<int64_t> ksize = ksize_opt.value().ToVector();
  std::vector<int64_t> strides = strides_opt.value().ToVector();
  std::vector<int64_t> pads = pads_opt.value().ToVector();

  auto AdjustVectorLength = [](std::vector<int64_t> &vec, const std::string &name) {
    if (!IsEqualToTargets(vec.size(), kDim1, kDim3)) {
      MS_EXCEPTION(ValueError) << "For 'MaxUnpool3d', the length of " << name << " should be 1 or 3, but got "
                               << vec.size();
    }
    if (vec.size() == 1) {
      vec.resize(kDim3, vec[0]);
    }
  };

  AdjustVectorLength(ksize, "kernel_size");
  AdjustVectorLength(strides, "stride");
  AdjustVectorLength(pads, "padding");

  auto size = out_shape.size();
  int64_t out_d = static_cast<int64_t>((out_shape[size - kDim3] - 1) * strides[kInputIndex0] -
                                       kDim2 * pads[kInputIndex0] + ksize[kInputIndex0]);
  int64_t out_h = static_cast<int64_t>((out_shape[size - kDim2] - 1) * strides[kInputIndex1] -
                                       kDim2 * pads[kInputIndex1] + ksize[kInputIndex1]);
  int64_t out_w = static_cast<int64_t>((out_shape[size - kDim1] - 1) * strides[kInputIndex2] -
                                       kDim2 * pads[kInputIndex2] + ksize[kInputIndex2]);

  out_shape[last_dim - kDim2] = out_d;
  out_shape[last_dim - kDim1] = out_h;
  out_shape[last_dim] = out_w;

  return {out_shape, strides};
}

ShapeArray MaxUnpool3DExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto in_shape = input_infos[kInputIndex0]->GetShape();

  if (input_infos[kInputIndex0]->IsDynamicRank()) {
    return {in_shape};
  }

  const int64_t input_num_dims = SizeToLong(in_shape.size());
  auto last_dim = input_num_dims - 1;
  ShapeVector out_shape = input_infos[kInputIndex0]->IsDynamic() ? input_infos[kInputIndex1]->GetShape() : in_shape;
  out_shape = input_infos[kInputIndex1]->IsDynamicRank() ? in_shape : out_shape;
  CheckAndConvertUtils::CheckInRange("dim of input", input_num_dims, kIncludeBoth, {4, 5}, primitive->name());

  if (input_infos[kInputIndex5]->IsNone()) {
    if (CheckMaxUnpool3DInputIsDynamic(input_infos)) {
      return {SetUnknownOutputShape(out_shape, last_dim)};
    }
    return {Cal3DOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4], out_shape,
                          last_dim, primitive)[0]};
  }

  auto output_shape_opt = input_infos[kInputIndex5]->GetArrayValue<int64_t>();
  size_t size = output_shape_opt->size();

  if (CheckMaxUnpool3DInputIsDynamic(input_infos)) {
    if (output_shape_opt.has_value()) {
      if (!IsEqualToTargets(size, kDim3, kDim5)) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                 << "', the length of output_size should be 3 or 5, but got " << size;
      }
      if (output_shape_opt.value().HasUnknownValue()) {
        out_shape = SetUnknownOutputShape(out_shape, last_dim);
      } else {
        auto output_shape_val = output_shape_opt.value();
        out_shape[last_dim - kDim2] = output_shape_val[size - kDim3];
        out_shape[last_dim - kDim1] = output_shape_val[size - kDim2];
        out_shape[last_dim] = output_shape_val[size - kDim1];
      }
    } else {
      out_shape = SetUnknownOutputShape(out_shape, last_dim);
    }
    return {out_shape};
  }

  if (output_shape_opt.has_value()) {
    if (!IsEqualToTargets(size, kDim3, kDim5)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the length of output_size should be 3 or 5, but got " << size;
    }
    if (output_shape_opt.value().HasUnknownValue()) {
      return {SetUnknownOutputShape(out_shape, last_dim)};
    }
    auto cal_out = Cal3DOutShape(input_infos[kInputIndex2], input_infos[kInputIndex3], input_infos[kInputIndex4],
                                 out_shape, last_dim, primitive);
    out_shape = cal_out[0];
    auto strides_out = cal_out[1];

    auto attr_output_shape = output_shape_opt.value();
    int64_t size_d = attr_output_shape[size - kDim3];
    int64_t size_h = attr_output_shape[size - kDim2];
    int64_t size_w = attr_output_shape[size - kDim1];

    auto min_max_size = [&](int64_t idx, const std::vector<int64_t> &strides) {
      int64_t min_size = out_shape[last_dim - idx] - strides[idx];
      int64_t max_size = out_shape[last_dim - idx] + strides[idx];
      return std::make_pair(min_size, max_size);
    };

    auto [min_size_d, max_size_d] = min_max_size(kDim2, strides_out);
    auto [min_size_h, max_size_h] = min_max_size(kDim1, strides_out);
    auto [min_size_w, max_size_w] = min_max_size(kDim0, strides_out);

    if (IsInRange(size_d, min_size_d, max_size_d) && IsInRange(size_h, min_size_h, max_size_h) &&
        IsInRange(size_w, min_size_w, max_size_w)) {
      out_shape[last_dim - kDim2] = size_d;
      out_shape[last_dim - kDim1] = size_h;
      out_shape[last_dim] = size_w;
    }
  } else {
    out_shape = SetUnknownOutputShape(out_shape, last_dim);
  }
  return {out_shape};
}

std::vector<TypeId> MaxUnpool3DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
