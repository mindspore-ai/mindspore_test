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

#include "kernel/ascend/pyboost/customize/mse_loss_grad_ext.h"
#include <memory>
#include <unordered_map>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ccsrc/pyboost/auto_generate/broadcast_to.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MSELossGradExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                const TensorPtr &grad_output_tensor, const TensorPtr &input_tensor,
                                                const TensorPtr &target_tensor, const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "MSELossGradExt call start";
  OpRunner::InferOpOutput(op, grad_output_tensor, input_tensor, target_tensor, reduction);

  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  const std::vector<int64_t> &expand_shape = op->output_value_simple_info()->shape_vector_[kIndex0];

  const ValueTuplePtr &expand_shape_ptr = ops::ConvertShapeVectorToValueTuple(expand_shape);
  MS_EXCEPTION_IF_NULL(expand_shape_ptr);

  TensorPtr input_tensor_bd = input_tensor;
  TensorPtr target_tensor_bd = target_tensor;

  const std::vector<int64_t> &input_shape = input_tensor->shape();
  if (input_shape != expand_shape) {
    const auto &broadcast_to_op =
      CREATE_PYBOOST_OP(BroadcastTo, op->device_context()->device_context_key_.device_name_);
    input_tensor_bd = broadcast_to_op->Call(input_tensor, expand_shape_ptr);
  }
  const std::vector<int64_t> &target_shape = target_tensor->shape();
  if (target_shape != expand_shape) {
    const auto &broadcast_to_op =
      CREATE_PYBOOST_OP(BroadcastTo, op->device_context()->device_context_key_.device_name_);
    target_tensor_bd = broadcast_to_op->Call(target_tensor, expand_shape_ptr);
  }

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), grad_output_tensor, input_tensor_bd,
                                target_tensor_bd);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, grad_output_tensor, input_tensor_bd, target_tensor_bd, reduction_value]() {
      MS_LOG(DEBUG) << "Run device task MSELossGradExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, grad_output_tensor, input_tensor_bd, target_tensor_bd);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnMseLossBackward, device_context, op->stream_id(), grad_output_tensor, input_tensor_bd,
                   target_tensor_bd, reduction_value, outputs[0]);
      MS_LOG(DEBUG) << "Run device task MSELossGradExt end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
