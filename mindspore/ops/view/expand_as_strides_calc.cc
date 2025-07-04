/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "view/expand_as_strides_calc.h"
#include <memory>
#include <string>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_op_name.h"
#include "view/broadcast_to_strides_calc.h"

namespace mindspore::ops {
constexpr size_t kExpandAsInputsNum = 2;

TensorStorageInfoPtrList ExpandAsCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto self_tensor = inputs[0]->cast<tensor::TensorPtr>();
  auto other_tensor = inputs[1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(other_tensor);
  return BroadCastToProcess(prim, self_tensor, other_tensor->shape());
}

REG_VIEW_STRIDES_CALC_FUN(ExpandAs, ExpandAsCalc);
}  // namespace mindspore::ops
