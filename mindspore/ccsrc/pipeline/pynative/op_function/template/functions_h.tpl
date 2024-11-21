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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GENERATE_FUNCTIONS_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GENERATE_FUNCTIONS_H_

#include <optional>
#include "ir/base_tensor.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using BaseTensorPtr = std::shared_ptr<tensor::BaseTensor>;
using CloneFunc = void (*)(const OpPtr &inplace_op, const PrimitivePtr &prim, const std::string &device_target,
                           ValuePtrList &&inputs);

void BACKEND_EXPORT RegisterCloneFunc(const CloneFunc &clone_func);
const CloneFunc& GetCloneFunc();

${op_call_with_grad}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GENERATE_FUNCTIONS_H_
