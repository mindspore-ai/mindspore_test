/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/utils/tensor_utils.h"
#include "pyboost/functions/auto_grad_guard.h"
#include "pynative/utils/pynative_utils.h"
#include "pynative/forward/pyboost/forward_task.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"

namespace mindspore::pynative {
static const char *OP_NAME = "Detach";
py::object PYNATIVE_EXPORT PyboostDetach(const py::object &input) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, OP_NAME, false,
                                     true);
  if (!tensor::IsTensorPy(input)) {
    MS_EXCEPTION(TypeError) << OP_NAME << " input is not a tensor";
  }
  PyNativeAlgo::PyBoost::MarkSideEffect(input.ptr());
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, 1, true>();
  auto promises = tensor::TransformPromise(py_output);
  const auto input_value = tensor::ConvertToValue(input);
  DispatchOp(std::make_shared<PassthroughFrontendTask>(
    [input_value, promises]() {
      const auto &input_tensor = PyNativeAlgo::Common::StubNodeToTensor(input_value);
      if (device::IsAscendDeviceType(input_tensor->device_address()->GetDeviceType())) {
        kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, device::DeviceType::kAscend));
      }
      const auto device_context =
        runtime::OpRunner::GetDeviceContext(kernel::pyboost::OpRunStatus::Get().device_target());
      kernel::pyboost::PyBoostUtils::PrepareOpInputs(
        device_context, device_context->device_res_manager_->GetCurrentStreamId(), input_tensor);
      auto output = std::make_shared<tensor::Tensor>(*input_tensor);
      output->set_auto_grad_meta_data(nullptr);
      // Async
      kernel::pyboost::PyBoostUtils::DispatchRun(
        std::make_shared<runtime::PyBoostDeviceTask>([input_tensor, device_context]() {
          MS_LOG(DEBUG) << "Run detach malloc op inputs start";
          // Malloc for input tensors
          kernel::pyboost::PyBoostUtils::MallocOpInputs(device_context, input_tensor);
          MS_LOG(DEBUG) << "Run device task Baddbmm end";
        }));

      tensor::SetPromise(promises, output);
    },
    [promises]() { tensor::SetException(promises); }));
  return py::reinterpret_steal<py::object>(tensor::TransformOutput(py_output));
}

void RegisterDetachFunction(py::module *m) { m->def("pyboost_detach", &PyboostDetach, OP_NAME); }
}  // namespace mindspore::pynative
