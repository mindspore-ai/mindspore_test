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

#include "kernel/cpu/pyboost/customize/empty.h"
#include <memory>
#include <string>
#include <vector>
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
ShapeVector GetShape(const ValueTuplePtr &shape) {
  ShapeVector output_shape;
  for (size_t i = 0; i < shape->size(); ++i) {
    int64_t shape_i = std::static_pointer_cast<Int64Imm>((*shape)[i])->value();
    output_shape.push_back(shape_i);
  }
  return output_shape;
}

TypeId GetDataType(const std::optional<Int64ImmPtr> &dtype) {
  // default type: float32
  TypeId data_type = kNumberTypeFloat32;
  if (dtype.has_value()) {
    data_type = static_cast<TypeId>(GetValue<int64_t>(dtype.value()));
    MS_LOG(DEBUG) << "dtype is not None, output tensor's dtype will be set to " << TypeIdToString(data_type);
  }
  return data_type;
}

std::string GetEmptyDeviceName(const std::optional<Int64ImmPtr> &device) {
  std::string device_name = "Ascend";
  if (device.has_value()) {
    auto device_name_enum = GetValue<int64_t>(device.value());
    if (device_name_enum == DEVICE_ASCEND || device_name_enum == DEVICE_NPU_LOWER) {
      device_name = "Ascend";
    } else if (device_name_enum == DEVICE_CPU || device_name_enum == DEVICE_CPU_LOWER) {
      device_name = "CPU";
    } else {
      MS_LOG(EXCEPTION) << "Only support ['CPU', 'Ascend', 'cpu', 'npu'] for device";
    }
  }
  MS_LOG(DEBUG) << "Using '" << device_name << "' as the device";
  return device_name;
}
}  // namespace

tensor::TensorPtr EmptyCPUCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &size,
                                    const std::optional<Int64ImmPtr> &dtype, const std::optional<Int64ImmPtr> &device) {
  MS_LOG(DEBUG) << "Call Empty start";
  ShapeVector output_shape = GetShape(size);
  TypeId data_type = GetDataType(dtype);
  std::string device_name = GetEmptyDeviceName(device);

  auto device_ctx = runtime::OpRunner::GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);

  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(data_type, output_shape, &outputs);
  PyBoostUtils::PrepareOpOutputs(device_ctx, op->stream_id(), outputs);
  op->set_outputs(outputs);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, device_ctx]() {
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_ctx, outputs);
    MS_LOG(DEBUG) << "Run device task Empty end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
