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
#ifndef MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

template <size_t N>
class CustomAclnnKernelMod : public AclnnKernelMod {
 public:
  explicit CustomAclnnKernelMod(std::string op_type) : AclnnKernelMod(std::move(op_type)) {}
  ~CustomAclnnKernelMod() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    const auto &res_tuple = GetKernelTuple<N>(inputs, outputs);
    std::apply([this](const auto &... args) { GetWorkspaceForResize(args...); }, res_tuple);
  }
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    CallRun(stream_ptr, workspace, inputs, outputs);
    return true;
  }

 private:
  template <typename... Ts>
  void CallRun(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const std::vector<Ts> &... vecs) {
    const auto &res_tuple = GetKernelTuple<N>(vecs...);
    std::apply(
      [this, stream_ptr, &workspace](const auto &... args) { return this->RunOp(stream_ptr, workspace, args...); },
      res_tuple);
  }

  DEFINE_GET_WORKSPACE_FOR_RESIZE()
};

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_ACLNN_KERNEL_MOD_H_
