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

#include "kernel/ascend/pyboost/customize/randn.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandnAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &shape,
                                       const TensorPtr &seed, const TensorPtr &offset,
                                       const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, shape, seed, offset, dtype);
  auto [seed_imm, offset_imm] = UpdateGeneratorState(seed, offset);
  auto device_context = op->device_context();
  auto outputs = op->outputs();
  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, seed_imm, offset_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors

    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    constexpr float mean = 0.0;
    constexpr float std = 1.0;
    LAUNCH_ACLNN(aclnnInplaceNormal, device_context, op->stream_id(), outputs[kIndex0], mean, std, seed_imm,
                 offset_imm);
  }));
  return outputs[kIndex0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
