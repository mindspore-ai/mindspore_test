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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CONV3D_EXT_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CONV3D_EXT_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace conv3d_ext {

class Conv3DExtAscend : public AclnnKernelMod {
 public:
  Conv3DExtAscend() : AclnnKernelMod(std::move("aclnnConvolution")) {}
  ~Conv3DExtAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape);
  TensorStorageInfoPtr CreateTensorStorageInfoPtr(const std::vector<int64_t> &new_shape,
                                                  const TensorStorageInfoPtr &old_tensor_storage_info);
  template <typename T>
  void SetTensorStorageInfo(T kernel_tensor, ShapeVector shape);
  template <typename T>
  void SetBackTensorStorageInfo(T kernel_tensor, ShapeVector shape);

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_{0};
  bool transposed_{false};
  std::vector<int64_t> output_padding_ = {0, 0, 0};
  std::shared_ptr<KernelTensor> input_kernel_tensor_;
  std::shared_ptr<KernelTensor> output_kernel_tensor_;
  bool _is_batchify{true};
  ShapeVector expand_out_shape_{};
};
}  // namespace conv3d_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CONV3D_EXT_ACLNN_KERNEL_MOD_H_
