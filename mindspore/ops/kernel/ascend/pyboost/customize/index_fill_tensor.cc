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

#include "kernel/ascend/pyboost/customize/index_fill_tensor.h"
#include <memory>
#include <string>

#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::BaseTensorPtr IndexFillTensorAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                                     const Int64ImmPtr &dim, const BaseTensorPtr &index,
                                                     const BaseTensorPtr &value) {
  OpRunner::InferOpOutput(op, input, dim, index, value);
  auto value_scalar = CreateValueFromTensor(value->cast<TensorPtr>())->cast<ScalarPtr>();
  auto dim_imm = GetValue<int64_t>(dim);
  std::vector<int64_t> index_vector;
  index->data_sync();
  TypeId tensor_type_id = static_cast<TypeId>(index->data_type_c());
  size_t elem_num = index->DataSize();
  if (tensor_type_id == TypeId::kNumberTypeInt64) {
    int64_t *elem_ptr = static_cast<int64_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt32) {
    int32_t *elem_ptr = static_cast<int32_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt16) {
    int16_t *elem_ptr = static_cast<int16_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeInt8) {
    int8_t *elem_ptr = static_cast<int8_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else if (tensor_type_id == TypeId::kNumberTypeUInt8) {
    uint8_t *elem_ptr = static_cast<uint8_t *>(index->data_c());
    for (size_t i = 0; i < elem_num; i++) {
      index_vector.push_back(elem_ptr[i]);
    }
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type for index conversion";
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input, dim_imm, index_vector, value_scalar]() {
      MS_LOG(DEBUG) << "Run device task "
                    << " end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << "Run device task "
                    << " start";
      LAUNCH_ACLNN(aclnnIndexFillTensor, device_context, op->stream_id(), input, dim_imm, index_vector, value_scalar,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task "
                    << " end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
