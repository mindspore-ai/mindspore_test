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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/common/pyboost/op_runner.h"
#include "mindspore/ops/kernel/ascend/opapi/aclnn_kernel_utils.h"
#include "mindspore/ops/kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class CustomAclnnPyboostKernelModBase {
 public:
  explicit CustomAclnnPyboostKernelModBase(std::string &&op_type) : op_type_(std::move(op_type)) {}
  ~CustomAclnnPyboostKernelModBase() = default;
  virtual bool Launch(const std::vector<tensor::BaseTensorPtr> &inputs,
                      const std::vector<tensor::BaseTensorPtr> &outputs,
                      const std::shared_ptr<pyboost::OpRunner> &op) = 0;
  std::string op_type_;
};

template <size_t N>
class CustomAclnnPyboostKernelMod : public CustomAclnnPyboostKernelModBase {
 public:
  explicit CustomAclnnPyboostKernelMod(std::string op_type) : CustomAclnnPyboostKernelModBase(std::move(op_type)) {}
  ~CustomAclnnPyboostKernelMod() = default;
  bool Launch(const std::vector<tensor::BaseTensorPtr> &inputs, const std::vector<tensor::BaseTensorPtr> &outputs,
              const std::shared_ptr<pyboost::OpRunner> &op) override {
    CallRun(op, inputs, outputs);
    return true;
  }

 private:
  template <typename... Ts>
  void CallRun(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<Ts> &... vecs) {
    MS_EXCEPTION_IF_NULL(op);
    auto stream_id = op->stream_id();
    auto device_context = op->device_context();
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
    auto aclnn_name = op_type_;
    auto res_tuple = GetKernelTuple<N>(vecs...);
    auto update_func = std::function<void()>(nullptr);
    auto [ws_size, executor_handle, release_function] =
      std::apply([&aclnn_name](const auto &... args) { return GEN_EXECUTOR(aclnn_name, args...); }, res_tuple);
    if (ws_size == 0) {
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, nullptr, 0, executor_handle, stream_ptr, release_function,
                             update_func);
    } else {
      auto workspace_device_address =
        runtime::DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(device_context, stream_id, ws_size);
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, workspace_device_address->GetMutablePtr(), ws_size,
                             executor_handle, stream_ptr, release_function, update_func);
    }
  }
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_KERNEL_H_
