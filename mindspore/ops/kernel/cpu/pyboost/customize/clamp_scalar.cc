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

#include "mindspore/ops/kernel/cpu/pyboost/customize/clamp_scalar.h"
#include "kernel/cpu/cpu_kernel.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "mindspore/ccsrc/pyboost/customize/op_common.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ClampScalarCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                              const std::optional<ScalarPtr> &min,
                                              const std::optional<ScalarPtr> &max) {
  return ClampScalarCustomizeCall(op, x_tensor, min, max, "CPU");
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
