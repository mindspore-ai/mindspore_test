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
#include "pynative/op_function/auto_generate/pyboost_api.h"
#include "pynative/op_function/auto_generate/pyboost_core.h"
#include "include/common/pybind_api/api_register.h"
#include "pynative/op_function/converter.h"
#include "op_def/auto_generate/gen_ops_def.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"

namespace mindspore::pynative {
${pyboost_op_base_body}
}// namespace mindspore::pynative
