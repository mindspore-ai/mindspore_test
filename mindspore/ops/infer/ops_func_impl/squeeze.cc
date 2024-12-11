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
ShapeVector ArrayToMultiplesVector(const ArrayValue<int64_t> &array_value) {
  auto len = array_value.size();
  ShapeVector multiples_vec;
  multiples_vec.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    if (array_value.IsValueUnknown(i)) {
      multiples_vec.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }
    multiples_vec.push_back(array_value[i]);
  }
  return multiples_vec;
}
}  // namespace
constexpr auto kSqueezeDim = 1;
BaseShapePtr SqueezeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto in_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto dim = GetArrayValue<int64_t>(input_args[kInputIndex1]);
  std::vector<int64_t> ret_shape;

  if (MS_UNLIKELY(IsDynamicRank(in_shape))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  auto dim_array = dim.value();
  auto dims = ArrayToMultiplesVector(dim_array);
  if (dims.empty()) {
    for (size_t i = 0; i < in_shape.size(); i++) {
      if (in_shape[i] != kSqueezeDim) {
        ret_shape.push_back(in_shape[i]);
      }
    }
  } else {
    auto rank = SizeToLong(in_shape.size());
    for (auto &item : dims) {
      CheckAndConvertUtils::CheckInRange<int64_t>("element or value of dim", item, kIncludeLeft, {-rank, rank},
                                                  "Squeeze");
    }
    for (int64_t i = 0; i < rank; i++) {
      auto it = std::find(dims.begin(), dims.end(), i);
      auto it2 = std::find(dims.begin(), dims.end(), i - rank);
      if ((it == dims.end() && it2 == dims.end()) || in_shape[i] != kSqueezeDim) {
        ret_shape.push_back(in_shape[LongToSize(i)]);
      }
    }
  }
  return std::make_shared<abstract::TensorShape>(ret_shape);
}

TypePtr SqueezeFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
