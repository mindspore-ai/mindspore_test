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

#include "infer/ops_func_impl/squeeze.h"
#include <utility>
#include <memory>
#include <vector>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"

#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "kernel/cpu/nnacl/op_base.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> CalRealDim(const std::vector<int64_t> &axes, const size_t &dim_size) {
  auto size = SizeToLong(dim_size);
  std::set<int64_t> seen_axes;
  std::vector<int64_t> real_axes;
  for (const auto &axis : axes) {
    // range validation
    int64_t real_axis = 0;
    if (size * -1 <= axis && axis < size) {
      real_axis = axis < 0 ? axis + size : axis;
    } else {
      MS_EXCEPTION(ValueError) << "dim value error. dim:" << axis << ", dim value should be in [" << -size << ", "
                               << size << ").";
    }
    // duplication validation
    if (seen_axes.find(real_axis) != seen_axes.end()) {
      MS_EXCEPTION(ValueError) << "dim value error. duplicate dim:" << real_axis;
    }
    seen_axes.insert(real_axis);
    real_axes.push_back(real_axis);
  }
  return real_axes;
}
}  // namespace
constexpr auto kSqueezeDim = 1;
BaseShapePtr SqueezeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto in_shape = input_args[kIndex0]->GetShape();
  auto in_shape_vec = in_shape->GetShapeVector();
  auto dim = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  std::vector<int64_t> ret_shape;

  if (MS_UNLIKELY(IsDynamicRank(in_shape_vec))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  if (!dim.has_value()) {
    // dim is None
    if (in_shape->IsDynamic()) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
    }
    for (size_t i = 0; i < in_shape_vec.size(); i++) {
      if (in_shape_vec[i] != kSqueezeDim) {
        ret_shape.push_back(in_shape_vec[i]);
      }
    }
    return std::make_shared<abstract::Shape>(ret_shape);
  }
  auto dim_array = dim.value();
  // if the dim has unknown value, the squeeze position could be any of the input dimensions.
  if (dim_array.HasUnknownValue()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  // All values of the dim are known
  std::vector<int64_t> dim_vec = dim_array.ToVector();
  std::vector<int64_t> real_dim_vec;
  auto ndim = in_shape_vec.size();
  real_dim_vec = CalRealDim(dim_vec, ndim);
  if (real_dim_vec.empty()) {
    for (size_t i = 0; i < in_shape_vec.size(); i++) {
      ret_shape.push_back(in_shape_vec[i]);
    }
    return std::make_shared<abstract::Shape>(ret_shape);
  }
  // if the squeeze dimension is the dynamic dim, return dynamic rank.
  if (std::any_of(real_dim_vec.begin(), real_dim_vec.end(),
                  [&](int64_t i) { return in_shape_vec[i] == abstract::Shape::kShapeDimAny; })) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  for (size_t i = 0; i < in_shape_vec.size(); i++) {
    auto it = std::find(real_dim_vec.begin(), real_dim_vec.end(), i);
    if (it == real_dim_vec.end() || in_shape_vec[i] != kSqueezeDim) {
      ret_shape.push_back(in_shape_vec[i]);
    }
  }
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr SqueezeFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
