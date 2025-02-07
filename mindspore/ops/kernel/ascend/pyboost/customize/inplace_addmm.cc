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

#include "kernel/ascend/pyboost/customize/inplace_addmm.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/ascend/pyboost/auto_generate/transpose.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplaceAddmmAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const BaseTensorPtr &input_tensor, const BaseTensorPtr &mat1_tensor,
                                                  const BaseTensorPtr &mat2_tensor, const ScalarPtr &beta,
                                                  const ScalarPtr &alpha) {
  OpRunner::InferOpOutput(op, input_tensor, mat1_tensor, mat2_tensor, beta, alpha);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, mat1_tensor, mat2_tensor);
  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, mat1_tensor, mat2_tensor, beta, alpha]() {
      MS_LOG(DEBUG) << "Run device task InplaceAddmm start";
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, mat1_tensor, mat2_tensor);
      // Malloc for output tensors

      // cubeMathType: 0 - KEEP_DTYPE, 1 - ALLOW_FP32_DOWN_PRECISION
      auto cube_math_type = GetCubeMathType(IsAllowMatmulHF32());
      LAUNCH_ACLNN(aclnnInplaceAddmm, device_context, op->stream_id(), input_tensor, mat1_tensor, mat2_tensor, beta,
                   alpha, cube_math_type);
      MS_LOG(DEBUG) << "Run device task InplaceAddmm end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
