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

#include "kernel/ascend/pyboost/customize/roi_align_ext.h"
#include <string>
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void RoiAlignExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                const BaseTensorPtr &boxes, const ValueTuplePtr &output_size,
                                const FP32ImmPtr &spatial_scale, const Int64ImmPtr &sampling_ratio,
                                const BoolImmPtr &aligned) {
  MS_LOG(DEBUG) << "Call start";
  OpRunner::InferOpOutput(op, input, boxes, output_size, spatial_scale, sampling_ratio, aligned);
  const auto output_size_val = ConvertValueTupleToVector<int64_t>(output_size);
  int64_t pooled_height_val = output_size_val[kDim0];
  int64_t pooled_width_val;
  if (output_size_val.size() == kDim2) {
    pooled_width_val = output_size_val[kDim1];
  } else {
    pooled_width_val = output_size_val[kDim0];
  }
  const auto spatial_scale_val = GetValue<float>(spatial_scale);
  auto sampling_ratio_val = GetValue<int64_t>(sampling_ratio);
  const auto aligned_val = GetValue<bool>(aligned);

  if (sampling_ratio_val < 0) {
    sampling_ratio_val = 0;
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, boxes);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input, boxes, pooled_height_val, pooled_width_val, spatial_scale_val, sampling_ratio_val, aligned_val]() {
      MS_LOG(DEBUG) << "Run device task RoiAlignExt start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input, boxes);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      LAUNCH_ACLNN(aclnnRoiAlignV2, device_context, op->stream_id(), input, boxes, pooled_height_val, pooled_width_val,
                   spatial_scale_val, sampling_ratio_val, aligned_val, outputs[0]);
      MS_LOG(DEBUG) << "Run device task RoiAlignExt end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
