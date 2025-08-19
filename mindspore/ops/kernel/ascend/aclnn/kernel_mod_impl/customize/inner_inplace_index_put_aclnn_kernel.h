/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_INNER_INPLACE_INDEX_PUT_ACLNN_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_INNER_INPLACE_INDEX_PUT_ACLNN_KERNEL_H_
#include <vector>
#include <utility>

#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
namespace inner_inplace_index_put {
class InnerInplaceIndexPutAscend : public AclnnKernelMod {
 public:
  InnerInplaceIndexPutAscend() : AclnnKernelMod(std::move("aclnnIndexPutImpl")) {}
  ~InnerInplaceIndexPutAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelTensor *> RemoveTrailingEmptyTensor(const std::vector<KernelTensor *> &indices);

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  bool accumulate_{false};
};
}  // namespace inner_inplace_index_put
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_OPAPI_ACLNN_INNER_INPLACE_INDEX_PUT_ACLNN_KERNEL_H_
