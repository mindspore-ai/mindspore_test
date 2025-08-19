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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V3_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V3_ACLNN_KERNEL_MOD_H_
#include <cstdint>
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace silent_check_v3 {

class SilentCheckV3Ascend : public AclnnKernelMod {
 public:
  SilentCheckV3Ascend() : AclnnKernelMod(std::move("aclnnSilentCheckV2")) {}
  ~SilentCheckV3Ascend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnSilentCheckV2, SilentCheckV3)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnInplaceCopy, InputGradCopy)

  pyfloat c_thresh_l1_{1000000.};
  pyfloat c_thresh_l2_{10000.};
  pyfloat beta1_{0.};
  int64_t npu_asd_detect_{1};

  std::vector<int64_t> dst_size_;
  std::vector<int64_t> dst_stride_;
  std::vector<int64_t> dst_offset_;
};
}  // namespace silent_check_v3
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SILENT_CHECK_V3_ACLNN_KERNEL_MOD_H_
