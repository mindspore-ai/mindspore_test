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
#include <list>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace custom {
constexpr size_t kWorkspaceIndex = 3;
constexpr size_t kReleaseFuncIndex = 2;

template <size_t N>
class CustomAclnnKernelMod : public AclnnKernelMod {
 public:
  explicit CustomAclnnKernelMod(std::string op_type) : AclnnKernelMod(std::move(op_type)) {}
  ~CustomAclnnKernelMod() = default;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    MS_LOG(DEBUG) << "Start get custom workspace info, op_type: " << op_type_;
    const auto &res_tuple = GetKernelTuple<N>(inputs, outputs);
    std::apply([this](const auto &... args) { GetWorkspaceForResize(args...); }, res_tuple);
    MS_LOG(DEBUG) << "End get custom workspace info, op_type: " << op_type_;
  }
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_LOG(DEBUG) << "Start launch custom, op_type: " << op_type_;
    CallRun(stream_ptr, workspace, inputs, outputs);
    MS_LOG(DEBUG) << "End launch custom, op_type: " << op_type_;
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

  template <typename... Args>
  void GetWorkspaceForResize(const Args &... args) {
    hash_id_ = device::ascend::AclnnHash(op_type_, args...);
    size_t cur_workspace = 0;
    std::optional<std::list<CacheTuple>::iterator> found_iter;
    if (auto iter = hash_map.find(hash_id_); iter != hash_map.end()) {
      found_iter = iter->second;
    }
    if (found_iter) {
      hash_cache.splice(hash_cache.begin(), hash_cache, *found_iter);
      cur_workspace = std::get<kWorkspaceIndex>(hash_cache.front());
    } else {
      auto [workspace, executor, cache, fail_cache] = GEN_CUSTOM_EXECUTOR_FOR_RESIZE(op_type_, args...);
      cur_workspace = workspace;
      if (!fail_cache) {
        hash_cache.emplace_front(hash_id_, executor, cache, workspace);
        hash_map[hash_id_] = hash_cache.begin();
        if (hash_cache.size() > capacity) {
          auto release_data = std::move(hash_cache.back());
          hash_map.erase(std::get<0>(release_data));
          hash_cache.pop_back();
          auto release_func = std::get<2>(release_data);
          release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
        }
      } else {
        hash_id_ = 0;
        cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
      }
    }
    if (cur_workspace != 0) {
      std::vector<size_t> workspace_size_list = {cur_workspace};
      SetWorkspaceSizeList(workspace_size_list);
    }
  }

  template <typename... Args>
  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &... args) {
    auto [executor, release_func] = GetExecutor(args...);
    if (workspace_size_list_.empty()) {
      RUN_CUSTOM_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);
    } else {
      if (workspace.empty()) {
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";
      }
      auto workspace_tensor = workspace[0];
      if (workspace_tensor->size() != workspace_size_list_[0]) {
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();
      }
      RUN_CUSTOM_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr,
                              release_func);
    }
  }

  template <typename... Args>
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor(const Args &... args) {
    if (hash_id_ == 0 || !hash_map.count(hash_id_)) {
      aclOpExecutor *executor;
      std::function<void()> release_func;
      std::tie(std::ignore, executor, std::ignore, release_func) = GEN_CUSTOM_EXECUTOR(op_type_, args...);
      return std::make_pair(executor, release_func);
    }
    const auto cur_run = *hash_map[hash_id_];
    UPDATE_TENSOR_FOR_LAUNCH(std::get<kReleaseFuncIndex>(cur_run), args...);
    const auto &executor = std::get<1>(cur_run);
    return std::make_pair(executor, nullptr);
  }
};

}  // namespace custom
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_ASCEND_OPAPI_CUSTOM_ACLNN_KERNEL_MOD_H_
