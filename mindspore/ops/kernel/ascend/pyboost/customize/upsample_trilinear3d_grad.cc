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

#include "kernel/ascend/pyboost/customize/upsample_trilinear3d_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindapi/base/types.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr pyfloat DEFAULT_SCALE_VALUE = 0.;
tensor::BaseTensorPtr UpsampleTrilinear3DGradAscendCall(
  const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context, const BaseTensorPtr &grad_out,
  const std::vector<int64_t> &input_size, const std::vector<int64_t> &output_size, const std::vector<pyfloat> &scales,
  const bool &align_corners, const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  double scales_d = scales[0];
  double scales_h = scales[1];
  double scales_w = scales[2];
  LAUNCH_ACLNN(aclnnUpsampleTrilinear3dBackward, device_context, op->stream_id(), grad_out, output_size, input_size,
               align_corners, scales_d, scales_h, scales_w, outputs[0]);
  MS_LOG(DEBUG) << "Call end";
  return outputs[0];
}
}  // namespace

tensor::BaseTensorPtr UpsampleTrilinear3DGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const BaseTensorPtr &gradout_tensor,
                                                             const ValueTuplePtr &input_size,
                                                             const std::optional<ValueTuplePtr> &output_size,
                                                             const std::optional<ValueTuplePtr> &scale_factors,
                                                             const BoolImmPtr &align_corners) {
  MS_LOG(DEBUG) << "UpsampleTrilinear3DGradAscendCustomize start";
  OpRunner::InferOpOutput(op, gradout_tensor, input_size, output_size, scale_factors, align_corners);

  auto input_size_vector = ConvertValueTupleToVector<int64_t>(input_size);

  std::vector<int64_t> output_size_vector{};
  std::vector<pyfloat> scales(kDim3, DEFAULT_SCALE_VALUE);
  if (output_size.has_value()) {
    output_size_vector = ConvertValueTupleToVector<int64_t>(output_size.value());
  } else if (scale_factors.has_value()) {
    scales = ConvertValueTupleToVector<pyfloat>(scale_factors.value());
    for (size_t i = 0; i < scales.size(); ++i) {
      output_size_vector.push_back(static_cast<int64_t>(input_size_vector[i + kDim2]) * scales[i]);
    }
  }

  auto align_corners_val = GetValue<bool>(align_corners);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), gradout_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, gradout_tensor, input_size_vector, output_size_vector, scales, align_corners_val]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, gradout_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      // Call aclnnUpsampleTrilinear3d
      UpsampleTrilinear3DGradAscendCall(op, device_context, gradout_tensor, input_size_vector, output_size_vector,
                                        scales, align_corners_val, outputs);
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
