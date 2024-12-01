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

#include "kernel/ascend/pyboost/customize/reshape_and_cache.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"
#include "plugin/device/ascend/kernel/internal/pyboost/acme_kernel_info.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeAndCacheAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &key,
                                                     const BaseTensorPtr &value, const BaseTensorPtr &key_cache,
                                                     const BaseTensorPtr &value_cache,
                                                     const BaseTensorPtr &slot_mapping) {
  OpRunner::InferOpOutput(op, key, value, key_cache, value_cache, slot_mapping);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), key, value, key_cache, value_cache,
                                slot_mapping);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  const auto &outputs = op->outputs();
  std::vector<BaseTensorPtr> inputs = {key, value, key_cache, value_cache, slot_mapping};
  std::shared_ptr<AcmeKernelInfo> kernel_info = nullptr;
  GET_ACMEKERNELINFO(kernel_info, "ReshapeAndCache", inputs, outputs);
  if (kernel_info == nullptr) {
    return nullptr;
  }
  auto tilingptr = kernel_info->GetOrGenerateTiling(inputs, outputs);
  if (tilingptr == nullptr) {
    return nullptr;
  }
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [kernel_info, tilingptr, op, key, value, key_cache, value_cache, slot_mapping]() {
      MS_LOG(DEBUG) << "Run device task Add start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, key, value, key_cache, value_cache, slot_mapping);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      kernel_info->Launch(device_context, tilingptr, {key, value, key_cache, value_cache, slot_mapping}, outputs,
                          op->stream_id());
      MS_LOG(DEBUG) << "Run device task Add end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
