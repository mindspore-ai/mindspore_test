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

#include <vector>
#include "infer/ops_func_impl/glu_grad.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace ops {
ShapeArray GluGradFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex1]->GetShape()};
}

std::vector<TypeId> GluGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex1]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
