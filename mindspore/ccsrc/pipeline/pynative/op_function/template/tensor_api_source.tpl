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

#include <memory>
#include "utils/ms_context.h"
#include "pybind_api/ir/arg_handler.h"
#include "pybind_api/ir/tensor_api/auto_generate/tensor_api.h"
#include "pybind_api/ir/tensor_register/auto_generate/tensor_func_utils.h"
#include "pipeline/pynative/op_function/converter.h"

namespace mindspore {
namespace tensor {

${tenosr_func_call_body}

}  // namespace tensor
}  // namespace mindspore