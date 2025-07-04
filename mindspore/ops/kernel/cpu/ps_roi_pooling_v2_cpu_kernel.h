/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_V2_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace ps_roi_pooling_v2_cpu {
class PSROIPoolingCpuKernelMod : public NativeCpuKernelMod {
 public:
  PSROIPoolingCpuKernelMod() = default;
  ~PSROIPoolingCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::vector<int64_t> input_shape;
  std::vector<int64_t> rois_shape;
  std::vector<int64_t> output_shape;

  int32_t batch_size_{-1};
  int32_t rois_num_{-1};
  int32_t output_n_{-1};
  float spatial_scale_{0.0};
  int32_t feature_channels_{-1};
  int32_t height_{-1};
  int32_t width_{-1};
  int32_t pooled_height_{-1};
  int32_t pooled_width_{-1};
  int32_t group_size_{-1};
  int32_t output_channels_{-1};

  TypeId data_type_id_{kNumberTypeFloat32};

  template <typename T>
  void PSROIPoolForward(size_t start, size_t end, const T *input, const T *roi_boxes, T *output_data);

  template <typename T>
  bool PSROIPoolingLauncher(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                            const int output_size);

  int ResizeCheckInputs(const std::vector<KernelTensor *> &inputs);
};
}  // namespace ps_roi_pooling_v2_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PS_ROI_POOLING_CPU_KERNEL_H_
