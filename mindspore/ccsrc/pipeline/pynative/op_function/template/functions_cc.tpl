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

#include <string>
#include "utils/ms_utils.h"
#include "mindspore/ops/kernel/functions/base.h"
#include "mindspore/ops/kernel/functions/auto_grad_reg.h"
#include "mindspore/ops/kernel/functions/auto_grad_guard.h"
#include "mindspore/ops/kernel/functions/auto_generate/functions.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "op_def/auto_generate/gen_ops_def.h"
${pyboost_op_header_include}

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
inline const std::string &GetDeviceTarget() { return OpGlobalStatus::Get().device_target(); }

using BaseTensorPtr = std::shared_ptr<tensor::BaseTensor>;
}

${op_call_with_grad}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
