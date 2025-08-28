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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_DEQUANT_SWIGLU_QUANT_ACLNN_KERNEL_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_DEQUANT_SWIGLU_QUANT_ACLNN_KERNEL_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
class DequantSwigluQuantAclnnKernelMod : public AclnnKernelMod {
 public:
  DequantSwigluQuantAclnnKernelMod() : AclnnKernelMod(std::move("aclnnDequantSwigluQuant")) {}
  ~DequantSwigluQuantAclnnKernelMod() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  enum DequantSwigluQuantInputIndex : size_t {
    kDequantSwigluQuantXIndex = 0,
    kDequantSwigluQuantWeightScaleIndex = 1,
    kDequantSwigluQuantActivationScaleIndex = 2,
    kDequantSwigluQuantBiasIndex = 3,
    kDequantSwigluQuantQuantScaleIndex = 4,
    kDequantSwigluQuantQuantOffsetIndex = 5,
    kDequantSwigluQuantGroupIndexIndex = 6,
    kDequantSwigluQuantActivateLeftIndex = 7,
    kDequantSwigluQuantQuantModeIndex = 8,
    kDequantSwigluQuantInputsNum = 9
  };
  bool activate_left_;
  std::string quant_mode_str_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_ACLNN_DEQUANT_SWIGLU_QUANT_ACLNN_KERNEL_H_
