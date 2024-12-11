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

#include "mindspore/ops/kernel/functions/base.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
OperatorType GetOpTypeFromOpdef(const ops::OpDef &op_def) {
  if (op_def.is_view_) {
    return OperatorType::kViewOp;
  }
  if (op_def.returns_[kIndex0].inplace_input_index_ == 0) {
    return OperatorType::kInplaceOp;
  }
  return OperatorType::kDefault;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
