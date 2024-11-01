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

#include "kernel/ascend/pyboost/customize/bincount_ext.h"
#include <memory>
#include "kernel/ascend/pyboost/auto_generate/min.h"
#include "kernel/ascend/pyboost/auto_generate/max.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
int64_t ConvertTensorToInt64(const BaseTensorPtr &tensor) {
  if (tensor == nullptr) {
    MS_LOG(EXCEPTION) << "Bincount ops receive null tensor.";
  }
  auto data_type = tensor->data_type_c();
  switch (data_type) {
    case kNumberTypeInt8:
      return static_cast<int64_t>(*(static_cast<int8_t *>(tensor->data_c())));
    case kNumberTypeInt16:
      return static_cast<int64_t>(*(static_cast<int16_t *>(tensor->data_c())));
    case kNumberTypeInt32:
      return static_cast<int64_t>(*(static_cast<int32_t *>(tensor->data_c())));
    case kNumberTypeInt64:
      return static_cast<int64_t>(*(static_cast<int64_t *>(tensor->data_c())));
    case kNumberTypeUInt8:
      return static_cast<int64_t>(*(static_cast<uint8_t *>(tensor->data_c())));
    default:
      MS_LOG(EXCEPTION) << "Unsupported input data type: " << data_type;
  }
  return 0;
}
}  // namespace

tensor::BaseTensorPtr BincountExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                                 const std::optional<BaseTensorPtr> &weight_tensor,
                                                 const Int64ImmPtr &min_length) {
  auto output_shape = GetValue<int64_t>(min_length);

  // Check if null tensor
  if (!(input_tensor->DataDim() == 1 && input_tensor->DataSize() == 0)) {
    auto min_op = CREATE_PYBOOST_OP(Min, op->device_context()->device_context_key_.device_name_);
    auto min_tensor = min_op->Call(input_tensor);
    auto max_op = CREATE_PYBOOST_OP(Max, op->device_context()->device_context_key_.device_name_);
    auto max_tensor = max_op->Call(input_tensor);

    // Get min value in input tensors
    min_tensor->data_sync();
    auto min_value = ConvertTensorToInt64(min_tensor);
    if (min_value < 0) {
      MS_LOG(EXCEPTION) << "Bincount only supports non-negative input values.";
    }
    // Get max value in input tensors and compare it  with min_length
    max_tensor->data_sync();
    auto max_value = ConvertTensorToInt64(max_tensor);
    output_shape = (max_value < output_shape) ? output_shape : (max_value + 1);
  }

  const auto output_shape_ptr = std::make_shared<Int64Imm>(output_shape);
  OpRunner::InferOpOutput(op, input_tensor, weight_tensor, output_shape_ptr);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, weight_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // update output shape
  auto output_real = ShapeVector({output_shape});
  op->UpdateOutputShape(op->output(kIndex0), output_real);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, weight_tensor,
                                                                          output_shape]() {
    MS_LOG(DEBUG) << "Run device task bincount start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, weight_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnBincount, device_context, op->stream_id(), input_tensor, weight_tensor, output_shape, outputs[0]);
    MS_LOG(DEBUG) << "Run device task bincount end";
  }));

  // update simple infer shape
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real};

  return op->output(kIndex0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
