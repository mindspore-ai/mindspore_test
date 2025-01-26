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
#include "pynative/op_function/auto_generate/pyboost_functions.h"
#include "include/common/pybind_api/api_register.h"
#include "pynative/pynative_execute.h"
#include "pynative/grad/grad_utils.h"
#include "pynative/pynative_utils.h"
#include "pynative/op_function/converter.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/utils/tensor_utils.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/predict_out_type_map.h"
#include "pynative/forward/forward_task.h"
#include "pipeline/jit/trace/trace_recorder.h"
#include "op_def/auto_generate/gen_ops_def.h"
#include "pynative/op_function/comm_handle_py.h"
#include "mindspore/ccsrc/pynative/op_function/auto_generate/tensor_func_utils.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"
${ops_inc}
${include_op_header}

namespace mindspore::pynative {
AsyncStatus GetAsyncStatus() {
  const auto &op_status = kernel::pyboost::OpRunStatus::Get().op_status();
  AsyncStatus status = {
    op_status.disable_mix_precision,
    op_status.is_jit_compiling,
    op_status.custom_bprop_cell_count,
  };
  return status;
}

${function_body}

${register_function_body}

${function_class_register}
}// namespace mindspore::pynative
