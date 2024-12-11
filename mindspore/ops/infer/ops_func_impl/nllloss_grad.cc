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

#include "infer/ops_func_impl/nllloss_grad.h"
#include <map>
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "abstract/dshape.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray NLLLossGradFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kInputIndex0]->GetShape();
  return {input_shape};
}
std::vector<TypeId> NLLLossGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto input_data_type = input_infos[kInputIndex0]->GetType();
  return {input_data_type};
}
}  // namespace ops
}  // namespace mindspore
