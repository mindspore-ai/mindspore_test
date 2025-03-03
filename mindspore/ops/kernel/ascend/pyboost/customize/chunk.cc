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
#include "kernel/ascend/pyboost/customize/chunk.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> ChunkAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                    const Int64ImmPtr &chunks, const Int64ImmPtr &dim) {
  MS_LOG(DEBUG) << "Chunk Call start";
  OpRunner::InferOpOutput(op, input_tensor, chunks, dim);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  const auto &input_shape = input_tensor->shape();
  auto chunks_value = chunks->value();
  auto dim_value = dim->value();
  if (dim_value < 0) {
    dim_value += SizeToLong(input_shape.size());
  }

  auto dim_size = input_shape.at(dim_value);
  auto split_size = (dim_size + chunks_value - 1) / chunks_value;
  // if true, use aclnnSplitWithSize, else aclnnSplitTensor
  bool flag = split_size == 0 && dim_size == 0;

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, flag, split_size,
                                                                          chunks_value, dim_value]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Launch Aclnn
    if (flag) {
      const std::vector<int64_t> split_sizes(chunks_value, 0);
      LAUNCH_ACLNN(aclnnSplitWithSize, device_context, op->stream_id(), input_tensor, split_sizes, dim_value, outputs);
    } else {
      LAUNCH_ACLNN(aclnnSplitTensor, device_context, op->stream_id(), input_tensor, split_size, dim_value, outputs);
    }

    MS_LOG(DEBUG) << "Chunk Launch end";
  }));

  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
