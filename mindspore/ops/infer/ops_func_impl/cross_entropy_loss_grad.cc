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

#include "infer/ops_func_impl/cross_entropy_loss_grad.h"
#include <string>
#include <set>
#include <algorithm>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray CrossEntropyLossGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto &log_prob = input_infos[kIndex1];
  auto log_prob_shape = log_prob->GetShape();
  return {log_prob_shape};
}

std::vector<TypeId> CrossEntropyLossGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type};
}
}  // namespace ops
}  // namespace mindspore
