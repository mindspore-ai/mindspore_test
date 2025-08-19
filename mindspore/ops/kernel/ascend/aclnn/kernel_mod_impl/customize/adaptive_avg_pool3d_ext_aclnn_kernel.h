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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL3D_EXT_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL3D_EXT_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace adaptive_avg_pool3d_ext {

class AdaptiveAvgPool3DExtAclnnKernelMod : public AclnnKernelMod {
 public:
  AdaptiveAvgPool3DExtAclnnKernelMod() : AclnnKernelMod(std::move("aclnnAdaptiveAvgPool3d")) {}
  ~AdaptiveAvgPool3DExtAclnnKernelMod() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMean, MeanExt);
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnAdaptiveAvgPool3d, AdaptiveAvgPool3DExt);
  std::vector<int64_t> output_size_vector_;
  TypeId dtype_;
  std::vector<int64_t> axis_{-1, -2, -3};  // {-1, -2, -3} fixed axis dims for aclnnMean
  bool keep_dims_{true};                   // true fixed keep_dims for aclnnMean
};
}  // namespace adaptive_avg_pool3d_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ADAPTIVE_AVG_POOL3D_EXT_ACLNN_KERNEL_MOD_H_
