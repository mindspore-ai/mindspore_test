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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_VIEW_TRANSPOSEEXT_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_VIEW_TRANSPOSEEXT_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "ir/tensor_storage_info.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {

class TransposeExtView : public AclnnKernelMod {
 public:
  TransposeExtView() : AclnnKernelMod("InnerTransposeExtView") {}
  ~TransposeExtView() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 private:
  mindspore::TensorStorageInfoPtrList info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_VIEW_TRANSPOSEEXT_KERNEL_MOD_H_
