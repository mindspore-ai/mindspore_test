/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/tile.h"

#include <algorithm>
#include <memory>
#include "ir/functor.h"
#include "mindapi/base/shape_vector.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ops/op_def/op_name.h"
#include "kernel/cpu/nnacl/op_base.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "utils/log_adapter.h"
#include "ir/primitive.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/dtype/number.h"

namespace mindspore::ops {
namespace {
ShapeVector ToMultiplesVector(const ArrayValue<int64_t> &array_value) {
  auto len = array_value.size();
  ShapeVector multiples_vec;
  multiples_vec.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    if (array_value.IsValueUnknown(i)) {
      multiples_vec.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }

    if (array_value[i] < 0) {
      MS_EXCEPTION(ValueError) << "For 'Tile', 'dims' cannot contain negative integer numbers, but got "
                               << array_value[i] << " in " << i << "th.";
    }
    multiples_vec.push_back(array_value[i]);
  }

  return multiples_vec;
}
}  // namespace
void AdaptShapeAndMultipies(ShapeVector *shape, ShapeVector *dims) {
  MS_EXCEPTION_IF_NULL(shape);
  if (MS_UNLIKELY(IsDynamicRank(*shape))) {
    MS_LOG(INTERNAL_EXCEPTION) << "Shape should not be dynamic rank!";
  }
  MS_EXCEPTION_IF_NULL(dims);

  auto rank = shape->size();
  auto len = dims->size();
  if (len == rank) {
    return;
  }

  auto expect_len = std::max(rank, len);
  auto ExpandInHeadIfNeed = [](ShapeVector *vec, size_t length) -> void {
    if (vec->size() == length) {
      return;
    }

    auto offset = length - vec->size();
    ShapeVector res;
    vec->reserve(length);
    vec->insert(vec->begin(), offset, 1);
  };

  ExpandInHeadIfNeed(shape, expect_len);
  ExpandInHeadIfNeed(dims, expect_len);
}

ShapeArray TileFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kInputIndex0];
  auto x_shape = x->GetShape();
  if (MS_UNLIKELY(x->IsDynamicRank())) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  auto &dim = input_infos[kInputIndex1];
  if (MS_UNLIKELY(dim->IsSequence() && dim->IsDynamicSequence())) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  auto dim_array_opt = dim->GetArrayValue<int64_t>();
  MS_CHECK_VALUE(dim_array_opt.has_value(),
                 CheckAndConvertUtils::FormatCommMsg("For primitive[Tile], the dims must has value."));
  auto dim_array = dim_array_opt.value();
  auto dims = ToMultiplesVector(dim_array);

  AdaptShapeAndMultipies(&x_shape, &dims);
  auto adapted_rank = x_shape.size();
  ShapeVector inferred_shape;
  inferred_shape.reserve(adapted_rank);
  for (size_t i = 0; i < adapted_rank; ++i) {
    if (x_shape[i] == abstract::Shape::kShapeDimAny || dims[i] == abstract::Shape::kShapeDimAny) {
      inferred_shape.push_back(abstract::Shape::kShapeDimAny);
      continue;
    }

    inferred_shape.push_back(LongMulWithOverflowCheck(dims[i], x_shape[i]));
  }
  return {inferred_shape};
}

std::vector<TypeId> TileFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace mindspore::ops
