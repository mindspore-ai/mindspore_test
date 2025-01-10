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

#include "infer/ops_func_impl/inner_strided_slice.h"

#include <algorithm>
#include <bitset>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
struct InnerStridedSliceSliceInfo {
  std::vector<int64_t> slice_value;
  size_t length;
  bool is_value_unknown{false};
  bool is_rank_unknown{false};
  // when is_value_unknown is true, the valid_value_info is valid.
  // represent the situation where the value is unknown. e.g. value=(1, None, 1), valid_value_info=(true, false, true).
  // when valid_value_info[i] is false, value[i] is INT64_MAX ==> value[i] is invalid.
  std::vector<bool> valid_value_info;
};

int64_t GetSlicingLength(int64_t start_pos, int64_t end_pos, int64_t strides, int64_t x_dim) {
  int64_t slicing_length = 0;
  if (strides <= 0) {
    MS_EXCEPTION(ValueError) << "For 'InnerStridedSlice', input 'strides' must be positive.";
  }
  if (x_dim == abstract::Shape::kShapeDimAny) {
    return abstract::Shape::kShapeDimAny;
  }
  if ((start_pos < x_dim) && end_pos >= -x_dim) {
    if (-x_dim <= start_pos && start_pos < 0) {
      start_pos += x_dim;
    }
    if (start_pos < -x_dim) {
      start_pos = 0;
    }
    if (-x_dim <= end_pos && end_pos < 0) {
      end_pos += x_dim;
    }
    if (end_pos > x_dim) {
      end_pos = x_dim;
    }
    if (start_pos >= end_pos) {
      slicing_length = 0;
    } else {
      slicing_length = 1 + (end_pos - 1 - start_pos) / strides;
    }
  }
  return slicing_length;
}

InnerStridedSliceSliceInfo GetSliceInfo(const AbstractBasePtr &input_arg) {
  InnerStridedSliceSliceInfo slice_info;
  auto slice_shape = input_arg->GetShape();
  if (slice_shape->isa<abstract::DynamicSequenceShape>()) {
    slice_info.is_rank_unknown = true;
    return slice_info;
  }

  auto slice_array_opt = GetArrayValue<int64_t>(input_arg);
  if (!slice_array_opt.has_value()) {
    if (slice_shape->isa<abstract::SequenceShape>()) {
      auto seq_shape = slice_shape->cast<abstract::SequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(seq_shape);
      size_t slice_size = seq_shape->size();
      slice_info.is_value_unknown = true;
      slice_info.length = slice_size;
      // represent slice value is (None, None, ...)
      std::vector<bool> valid_info(slice_size, false);
      std::vector<int64_t> slice_v(slice_size, INT64_MAX);
      slice_info.valid_value_info = valid_info;
      slice_info.slice_value = slice_v;
      return slice_info;
    }
    slice_info.is_rank_unknown = true;
    return slice_info;
  }

  auto slice_array = slice_array_opt.value();
  if (!slice_array.HasUnknownValue()) {
    slice_info.slice_value = slice_array.ToVector();
    slice_info.length = slice_info.slice_value.size();
    std::vector<bool> valid_info(slice_info.length, true);
    slice_info.valid_value_info = valid_info;
    return slice_info;
  }

  slice_info.is_value_unknown = true;
  slice_info.length = slice_array.size();
  for (size_t i = 0; i < slice_array.size(); i++) {
    if (slice_array.IsValueUnknown(i)) {
      slice_info.valid_value_info.push_back(false);
      slice_info.slice_value.push_back(INT64_MAX);  // placeholder, invalid value
    } else {
      slice_info.valid_value_info.push_back(true);
      slice_info.slice_value.push_back(slice_array[i]);
    }
  }
  return slice_info;
}

ShapeVector DynamicComputeInferShape(const InnerStridedSliceSliceInfo &begin_info,
                                     const InnerStridedSliceSliceInfo &end_info,
                                     const InnerStridedSliceSliceInfo &strides_info, const ShapeVector &x_shape) {
  size_t slice_len = begin_info.length;
  size_t i = 0;
  size_t j = 0;
  ShapeVector infer_shape;
  size_t x_rank = x_shape.size();

  while (i < x_rank || j < slice_len) {
    int64_t slicing_length = -1;
    int64_t x_dim_size = x_shape[i];
    int64_t begin = 0;
    int64_t end = x_shape[i];
    int64_t stride = 1;
    if (j < slice_len) {
      begin = begin_info.slice_value[j];
      end = end_info.slice_value[j];
      stride = strides_info.slice_value[j];

      bool is_slice_valid_value =
        begin_info.valid_value_info[j] && end_info.valid_value_info[j] && strides_info.valid_value_info[j];
      if (is_slice_valid_value) {
        slicing_length = GetSlicingLength(begin, end, stride, x_dim_size);
      }
    } else {
      slicing_length = GetSlicingLength(begin, end, stride, x_dim_size);
    }
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return infer_shape;
}

ShapeVector ComputeInferShape(const ShapeVector &begin_v, const ShapeVector &end_v, const ShapeVector &strides_v,
                              const ShapeVector &x_shape) {
  auto slice_len = begin_v.size();
  size_t i = 0;
  size_t j = 0;
  int64_t start;
  int64_t finish;
  int64_t strides;
  ShapeVector infer_shape;
  size_t x_rank = x_shape.size();
  while (i < x_rank || j < slice_len) {
    int64_t x_dim_size = x_shape[i];
    if (j < slice_len) {
      start = begin_v[j];
      finish = end_v[j];
      strides = strides_v[j];
    } else {
      start = 0;
      finish = x_shape[i];
      strides = 1;
    }
    int64_t slicing_length = GetSlicingLength(start, finish, strides, x_dim_size);
    infer_shape.push_back(slicing_length);
    i += 1;
    j += 1;
  }
  return infer_shape;
}
}  // namespace

BaseShapePtr InnerStridedSliceFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  ShapeVector ret_in_shape;
  auto prim_name = primitive->name();
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (x_shape.size() == 0) {
    MS_EXCEPTION(TypeError) << "For 'InnerStridedSlice', input can not be a scalar.";
  }
  auto begin_info = GetSliceInfo(input_args[kIndex1]);
  auto end_info = GetSliceInfo(input_args[kIndex2]);
  auto strides_info = GetSliceInfo(input_args[kIndex3]);
  std::vector<bool> check_vec = {IsDynamicRank(x_shape), begin_info.is_rank_unknown, end_info.is_rank_unknown,
                                 strides_info.is_rank_unknown};
  if (std::any_of(check_vec.begin(), check_vec.end(), [](const bool &flag) { return flag; })) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }

  if (begin_info.length != strides_info.length || end_info.length != strides_info.length) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'begin', 'end' and 'strides' must have the same length, "
                             << "but got length of 'begin': " << begin_info.length << ", 'end': " << end_info.length
                             << ", 'strides': " << strides_info.length << ".";
  }

  bool slice_dynamic = false;
  if (begin_info.is_value_unknown || end_info.is_value_unknown || strides_info.is_value_unknown || IsDynamic(x_shape)) {
    slice_dynamic = true;
  }
  if (!slice_dynamic) {
    ret_in_shape = ComputeInferShape(begin_info.slice_value, end_info.slice_value, strides_info.slice_value, x_shape);
  } else {
    ret_in_shape = DynamicComputeInferShape(begin_info, end_info, strides_info, x_shape);
  }

  return std::make_shared<abstract::Shape>(ret_in_shape);
}

TypePtr InnerStridedSliceFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
