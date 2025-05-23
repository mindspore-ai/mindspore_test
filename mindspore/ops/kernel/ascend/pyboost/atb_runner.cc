/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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

#include "kernel/ascend/pyboost/atb_runner.h"
#include "atb/atb_infer.h"
#include "debug/profiler/profiler.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "runtime/pynative/op_executor.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore::kernel::pyboost {
MS_ATB_RUNNER_REG(add, atb::infer::ElewiseParam);
}  // namespace mindspore::kernel::pyboost
