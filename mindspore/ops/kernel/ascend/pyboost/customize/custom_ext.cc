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

#include "kernel/ascend/pyboost/customize/custom_ext.h"
#include <string>
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/ascend/opapi/aclnn/custom_aclnn_utils.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "runtime/pynative/op_runner.h"
#include "mindspore/ops/kernel/ascend/opapi/aclnn/custom_aclnn_utils.h"
#include "mindspore/ops/kernel/ascend/pyboost/customize/custom_launch_aclnn.h"

namespace mindspore::kernel::pyboost {

void LaunchCustomAclnn(const std::string &aclnn_name, const std::shared_ptr<OpRunner> &op,
                       const std::vector<ValuePtr> &inputs, const std::vector<BaseTensorPtr> &output_tensors) {
  MS_EXCEPTION_IF_NULL(op);
  MS_LOG(DEBUG) << "Run device task custom " << aclnn_name << " start";
  MS_VLOG(VL_CUSTOM_OP) << "Run device task custom " << aclnn_name << " start";

  auto arg_num = inputs.size() + output_tensors.size();
  auto kernel_mod = GetCustomAclnnPyboostKernelMod(aclnn_name, arg_num);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto stream_id = op->stream_id();
  auto device_context = op->device_context();
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);

  kernel_mod->Launch(inputs, output_tensors, op);
  auto sync = runtime::RuntimeConf::GetInstance()->launch_blocking();
  if (sync) {
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;
    }
  } else {
    std::vector<BaseTensorPtr> input_tensors;
    for (const auto &item : inputs) {
      (void)input_tensors.emplace_back(item->cast<BaseTensorPtr>());
    }
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, input_tensors,
                                                           output_tensors);
  }

  MS_LOG(DEBUG) << "Run device task custom " << aclnn_name << " end";
  MS_VLOG(VL_CUSTOM_OP) << "Run device task custom " << aclnn_name << " end";
}

std::vector<tensor::BaseTensorPtr> CustomExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
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
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, tensors_tensor_list_vector, tensors_tensor_list]() {
      auto primitive = op->primitive();
      auto aclnn_name = GetValue<std::string>(primitive->GetAttr("reg_op_name"));

      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, tensors_tensor_list_vector);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LaunchCustomAclnn(aclnn_name, op, tensors_tensor_list->value(), op->outputs());
    }));
  return op->outputs();
}

class CustomAclnnOp : public OpRunner {
 public:
  using OpRunner::OpRunner;
  ~CustomAclnnOp() = default;
};

void CustomLaunchAclnnImpl(const std::string &aclnn_name, const ValuePtrList &inputs,
                           const tensor::BaseTensorPtrList &outputs) {
  auto p = std::make_shared<Primitive>("CustomLaunchAclnn");
  auto op = std::make_shared<CustomAclnnOp>(p, runtime::OpRunner::GetDeviceContext("Ascend"));
  op->set_stream_id(PyBoostUtils::cur_stream_id());

  tensor::BaseTensorPtrList input_tensors;
  input_tensors.reserve(inputs.size());
  for (auto &inp : inputs) {
    if (inp->isa<tensor::BaseTensor>()) {
      (void)input_tensors.emplace_back(inp->cast<tensor::BaseTensorPtr>());
    }
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensors);

  for (auto &out : outputs) {
    // this is set in CreateTensor called by PyBoostUtils::InferOpOutput.
    out->set_need_pipeline_sync(true);
  }
  op->set_outputs(outputs);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), outputs);

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, inputs, input_tensors, outputs, aclnn_name]() {
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensors);
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LaunchCustomAclnn(aclnn_name, op, inputs, outputs);
    }));
}
}  // namespace mindspore::kernel::pyboost

namespace mindspore::custom {
void CustomLaunchAclnn(const std::string &aclnn_name, const ValuePtrList &inputs,
                       const tensor::BaseTensorPtrList &outputs) {
  return mindspore::kernel::pyboost::CustomLaunchAclnnImpl(aclnn_name, inputs, outputs);
}
}  // namespace mindspore::custom
