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

#include "kernel/ascend/pyboost/customize/custom.h"
#include <string>
#include "kernel/ascend/pyboost/auto_generate/custom.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "runtime/device/device_address_utils.h"
#include "mindspore/ops/kernel/ascend/opapi/aclnn/custom_aclnn_utils.h"

namespace mindspore::kernel::pyboost {
namespace {

void LaunchCustomAclnn(const std::string &aclnn_name, const std::shared_ptr<OpRunner> &op,
                       std::vector<BaseTensorPtr> input_tensors, std::vector<BaseTensorPtr> output_tensors) {
  MS_EXCEPTION_IF_NULL(op);
  MS_LOG(DEBUG) << "Run device task custom " << aclnn_name << " start";
  MS_VLOG(VL_CUSTOM_OP) << "Run device task custom " << aclnn_name << " start";

  auto arg_num = input_tensors.size() + output_tensors.size();
  auto kernel_mod = GetCustomAclnnPyboostKernelMod(aclnn_name, arg_num);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto stream_id = op->stream_id();
  auto device_context = op->device_context();
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);
  kernel_mod->Launch(input_tensors, output_tensors, op);
  static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (sync) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;
    }
  } else {
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, input_tensors,
                                                           output_tensors[0]);
  }

  MS_LOG(DEBUG) << "Run device task custom " << aclnn_name << " end";
  MS_VLOG(VL_CUSTOM_OP) << "Run device task custom " << aclnn_name << " end";
}
}  // namespace

tensor::BaseTensorPtr CustomAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                            const ValueTuplePtr &tensors_tensor_list) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(tensors_tensor_list);
  MS_LOG(DEBUG) << "Start custom ascend customize";
  MS_VLOG(VL_CUSTOM_OP) << "Start custom ascend customize";
  OpRunner::InferOpOutput(op, tensors_tensor_list);
  // ValueTuple to std::vector
  std::vector<BaseTensorPtr> tensors_tensor_list_vector = ConvertValueTupleToVector<BaseTensorPtr>(tensors_tensor_list);
  auto device_context = op->device_context();
  PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), tensors_tensor_list_vector);
  for (size_t i = 0; i < tensors_tensor_list_vector.size(); i++) {
    MS_VLOG(VL_CUSTOM_OP) << "debug tensor: " << tensors_tensor_list_vector[i]->ToStringRepr();
  }
  PyBoostUtils::PrepareOpOutputs(device_context, op->stream_id(), op->outputs());
  for (size_t i = 0; i < op->outputs().size(); i++) {
    MS_VLOG(VL_CUSTOM_OP) << "output debug tensor: " << op->outputs()[i]->ToStringRepr();
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_tensor_list_vector]() {
    auto primitive = op->primitive();
    auto aclnn_name = GetValue<std::string>(primitive->GetAttr("reg_op_name"));

    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, tensors_tensor_list_vector);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    LaunchCustomAclnn(aclnn_name, op, tensors_tensor_list_vector, op->outputs());
  }));
  return op->output(0);
}
}  // namespace mindspore::kernel::pyboost
