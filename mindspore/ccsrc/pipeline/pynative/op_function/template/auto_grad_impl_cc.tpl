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

#include "op_def/auto_generate/gen_ops_def.h"
#include "mindspore/ops/kernel/functions/auto_grad_reg.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/ops/kernel/common/pyboost/op_runner.h"
#include "mindspore/ops/kernel/functions/auto_grad_guard.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "pipeline/pynative/grad/grad_utils.h"
#include "pipeline/pynative/op_function/auto_grad_register.h"

namespace mindspore::pynative {
namespace {
inline AsyncStatus GetAsyncStatus() {
  const auto &op_status = kernel::pyboost::OpRunStatus::Get().op_status();
  AsyncStatus status = {
    op_status.disable_mix_precision,
    op_status.is_jit_compiling,
    op_status.custom_bprop_cell_count,
  };
  return status;
}

inline bool NeedAutoGrad() {
  MS_LOG(DEBUG) << "require grad " << kernel::pyboost::OpRunStatus::Get().RequireGrad();
  return kernel::pyboost::OpRunStatus::Get().RequireGrad();
}
}

${do_grad_op}

void OpsAutoGradImplRegister() {
  ${auto_grad_reg}
}
}  // namespace mindspore::pynative
