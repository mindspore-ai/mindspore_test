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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DENSE_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DENSE_ACLNN_KERNEL_MOD_H_

#include <vector>
#include <utility>
#include <memory>
#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace dense {
class DenseAclnnKernelMod : public AclnnKernelMod {
 public:
  DenseAclnnKernelMod() : AclnnKernelMod(std::move("aclnnMatmul")) {}
  ~DenseAclnnKernelMod() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnPermute, Transform)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnAddmm, Addmm)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMatmul, Matmul)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnAdd, Add)

  void SetFlatternNdLinearTensorStorageInfo(const KernelTensorPtr &new_tensor, const int &new_shape_first,
                                            const ShapeVector &shape);

  std::vector<int64_t> t_perm_{};
  KernelTensor w_t_tensor_;
  KernelTensor matmul_tensor_;
  ScalarPtr one_ = nullptr;
  int8_t cube_math_type_{0};
  std::shared_ptr<KernelTensor> input_kernel_tensor_;
  std::shared_ptr<KernelTensor> output_kernel_tensor_;
};
}  // namespace dense
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_DENSE_ACLNN_KERNEL_MOD_H_
