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

#ifndef MINDSPORE_CORE_OPS_OP_FUNC_IMPL_LOGSUMEXP_H
#define MINDSPORE_CORE_OPS_OP_FUNC_IMPL_LOGSUMEXP_H

#include <vector>
#include <set>
#include <memory>
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
class LogSumExpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };
  // For aclnn GetWorkspace
  std::set<int64_t> GetValueDependArgIndices() const override { return {kIndex1, kIndex2}; };
};
using LogSumExpFuncImplPtr = std::shared_ptr<LogSumExpFuncImpl>;
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_FUNC_IMPL_LOGSUMEXP_H
