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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_AVG_POOL2D_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_AVG_POOL2D_H_

#include <memory>
#include <set>
#include <vector>
#include "ir/dtype/number.h"
#include "mindapi/base/types.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API AvgPool2DFuncImpl final : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  int32_t CheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  ShapeArray InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const override;

  TypePtrList InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const override;

 private:
  const size_t no_batch_rank_ = 3;
  const size_t batch_rank_ = 4;
  const int64_t tuple_min_ele_num_ = 1;
  const int64_t tuple_max_ele_num_ = 2;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_AVG_POOL2D_H_
