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

#include "infer/ops_func_impl/inplace_index_put.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

ShapeArray InplaceIndexPutFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[kIndex0]);
  auto &indices = input_infos[kIndex1];
  ShapeVector output_shape;
  if (indices->IsSequence() && indices->IsDynamicSequence()) {
    MS_EXCEPTION(ValueError) << "For `InplaceIndexPut` op, 'indices' shape can not DynamicSequenceShape.";
  }
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> InplaceIndexPutFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[kIndex0]);
  return {input_infos[kIndex0]->GetType()};
}
REGISTER_SIMPLE_INFER(kNameInplaceIndexPut, InplaceIndexPutFuncImpl)
}  // namespace ops
}  // namespace mindspore
