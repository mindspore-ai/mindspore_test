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

#include <iterator>
#include <memory>
#include <string>
#include "ir/scalar.h"
#include "pipeline/pynative/op_function/converter.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/op_executor.h"
#include "include/common/utils/stub_tensor.h"
#include "pipeline/pynative/pynative_utils.h"
#include "op_def/auto_generate/gen_ops_def.h"
#include "pybind_api/ops/direct_ops.h"

namespace mindspore::pynative {

py::object EmptyLike(const py::list &args) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, "EmptyLike",
                                     false, true);
  static const TypePtr type = std::make_shared<TensorType>();
  auto stub_out = stub::MakeTopNode(type);
  auto stub_node = stub_out.second;

  static pynative::Converter converter(&ops::gEmptyLike);
  auto input_tensor = converter.ToTensor(args, kIndex0);
  auto dtype = converter.ToDtypeOptional(args, kIndex1);
  auto device = converter.ToStringOptional(args, kIndex2);

  MS_LOG(DEBUG) << "start EmptyLike";

  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [stub_node, input_tensor, dtype, device]() {
      auto tensor_ptr = pynative::PyNativeAlgo::Common::ConvertStubNodeToTensor(input_tensor, true, false);
      std::string device_name;
      if (device.has_value()) {
        device_name = device.value()->value();
        if (device_name != "CPU" && device_name != "Ascend") {
          MS_LOG(EXCEPTION) << "Only support ['CPU', 'Ascend'] for device, but get '" << device_name << "'";
        }
        MS_LOG(DEBUG) << "Using input device_name: " << device_name;
      } else {
        auto device_address = std::static_pointer_cast<device::DeviceAddress>(tensor_ptr->device_address());
        if (device_address != nullptr) {
          device_name = device_address->device_name();
          MS_LOG(DEBUG) << "Using input tensor device_name: " << device_name;
        } else {
          auto ms_context = MsContext::GetInstance();
          MS_EXCEPTION_IF_NULL(ms_context);
          device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
          MS_LOG(DEBUG) << "Using default device_name: " << device_name;
        }
      }

      auto device_ctx = runtime::OpRunner::GetDeviceContext(device_name);
      MS_EXCEPTION_IF_NULL(device_ctx);

      TypeId real_type;
      if (dtype.has_value()) {
        real_type = static_cast<TypeId>(dtype.value()->value());
        MS_LOG(DEBUG) << "dtype.has_value == True, input type: " << TypeIdToString(real_type);
      } else {
        real_type = tensor_ptr->data_type();
        MS_LOG(DEBUG) << "dtype.has_value == False, input tensor type: " << TypeIdToString(real_type);
      }
      auto output_shape = tensor_ptr->shape();

      auto value_simple_info = std::make_shared<ValueSimpleInfo>();
      value_simple_info->shape_vector_.push_back(output_shape);
      value_simple_info->dtype_vector_.push_back(TypeIdToType(real_type));
      value_simple_info->size_ = 1;
      stub_node->SetValueSimpleInfo(value_simple_info);

      std::vector<tensor::BaseTensorPtr> outputs;
      kernel::pyboost::PyBoostUtils::CreateOutputTensor(real_type, output_shape, &outputs);
      kernel::pyboost::PyBoostUtils::PrepareOpOutputs(device_ctx, 0, outputs);
      stub_node->SetValue(outputs[0]);

      auto fn = [device_ctx, outputs]() { kernel::pyboost::PyBoostUtils::MallocOpOutputs(device_ctx, outputs); };

      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughNoWaitDeviceTask>(fn));
      } else {
        fn();
      }
    },
    stub_node));
  MS_LOG(DEBUG) << "finish EmptyLike";

  return stub_out.first;
}
}  // namespace mindspore::pynative
