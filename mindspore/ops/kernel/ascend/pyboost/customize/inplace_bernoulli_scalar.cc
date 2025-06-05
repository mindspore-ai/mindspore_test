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

#include "kernel/ascend/pyboost/customize/inplace_bernoulli_scalar.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceBernoulliScalarAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                        const FP32ImmPtr &p, const TensorPtr &seed,
                                                        const TensorPtr &offset) {
  MS_LOG(DEBUG) << "Call InplaceBernoulliScalar start";
  auto p_scalar = p->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(p_scalar);
  auto [seed_imm, offset_imm] = UpdateGeneratorState(seed, offset);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  op->set_outputs({input});
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, p_scalar, seed_imm, offset_imm]() {
    MS_LOG(DEBUG) << "Run device task InplaceBernoulliScalar start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input);
    // Inplace output need be front
    LAUNCH_ACLNN(aclnnInplaceBernoulli, device_context, op->stream_id(), input, p_scalar, seed_imm, offset_imm);
    MS_LOG(DEBUG) << "Launch InplaceBernoulliScalar end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
