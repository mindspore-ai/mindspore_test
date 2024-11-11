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

#include "infer/ops_func_impl/log2.h"
#include <memory>
#include <unordered_set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
const std::unordered_set<TypeId> int_or_bool_set_log2 = {kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                                         kNumberTypeInt32, kNumberTypeInt64, kNumberTypeBool};
}

ShapeArray Log2FuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> Log2FuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  if (int_or_bool_set_log2.find(type) != int_or_bool_set_log2.end()) {
    return {kNumberTypeFloat32};
  }
  return {type};
}
}  // namespace ops
}  // namespace mindspore
