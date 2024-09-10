/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <utility>
#include <tuple>
#include <list>
#include <unordered_map>
#include <memory>
#include "runtime/device/device_address_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "transform/acl_ir/op_api_exec.h"
#include "transform/acl_ir/op_api_convert.h"

using ProcessCache = mindspore::transform::ProcessCache;
using CacheTuple = std::tuple<uint64_t, mindspore::transform::aclOpExecutor *, ProcessCache, size_t>;

#define DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, ws_ptr, ws_size, executor, stream, release_func, \
                               update_func)                                                                 \
  runtime::OpExecutor::DispatchLaunchTask([=]() {                                                           \
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,                                  \
                                       runtime::ProfilerEvent::kPyNativeLaunchTask, aclnn_name, false);     \
    MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                   \
    device_context->device_res_manager_->BindDeviceToCurrentThread(false);                                  \
    if (update_func != nullptr) {                                                                           \
      update_func();                                                                                        \
    }                                                                                                       \
    RUN_OP_API_ASYNC(aclnn_name, ws_ptr, ws_size, executor, stream, release_func);                          \
    MS_LOG(DEBUG) << "launch task end, " << aclnn_name;                                                     \
  });

#define GET_EXECUTOR_FOR_PYBOOST(aclnn_api, ...)                                                  \
  [](const std::string &api_str, const auto &... args) -> auto {                                  \
    uint64_t hash_id = mindspore::transform::AclnnHash(api_str, args...);                         \
    if (hash_id != 0 && hash_map_.count(hash_id) != 0) {                                          \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, hash_map_[hash_id]);                   \
      auto cur_run = hash_cache_.front();                                                         \
      const auto &ws_size = std::get<3>(cur_run);                                                 \
      const auto &executor = std::get<1>(cur_run);                                                \
      const auto &cache = std::get<2>(cur_run);                                                   \
      auto address_list = mindspore::transform::GetTensorAddress(args...);                        \
      std::function<void()> update_func = [cache, address_list]() -> void {                       \
        cache(transform::ProcessCacheType::kUpdateTensorAddress, address_list);                   \
      };                                                                                          \
      auto release_func = std::function<void()>(nullptr);                                         \
      return std::make_tuple(ws_size, executor, cache, release_func, update_func);                \
    } else {                                                                                      \
      MS_LOG(INFO) << "Api " << api_str << " miss cache, with hash id:" << hash_id;               \
      auto [ws_size, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(api_str, args...);    \
      auto update_func = std::function<void()>(nullptr);                                          \
      if (hash_id != 0 && !fail_cache) {                                                          \
        hash_cache_.emplace_front(hash_id, executor, cache, ws_size);                             \
        hash_map_[hash_id] = hash_cache_.begin();                                                 \
        if (hash_cache_.size() > capacity_) {                                                     \
          hash_map_.erase(std::get<0>(hash_cache_.back()));                                       \
          auto release_func = std::get<2>(hash_cache_.back());                                    \
          release_func(transform::ProcessCacheType::kReleaseParamsAndExecutor, {});               \
          hash_cache_.pop_back();                                                                 \
        }                                                                                         \
        auto release_func = std::function<void()>(nullptr);                                       \
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);              \
      } else {                                                                                    \
        std::function<void()> release_func = [cache]() -> void {                                  \
          cache(transform::ProcessCacheType::kReleaseParams, std::vector<std::vector<void *>>{}); \
        };                                                                                        \
        return std::make_tuple(ws_size, executor, cache, release_func, update_func);              \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  (aclnn_api, __VA_ARGS__)

#define LAUNCH_ACLNN(aclnn_api, device_context, stream_id, ...)                                                     \
  do {                                                                                                              \
    static auto simu = !common::GetEnv(kSimulationLevel).empty();                                                   \
    if (simu) {                                                                                                     \
      break;                                                                                                        \
    }                                                                                                               \
    static const std::string aclnn_name = #aclnn_api;                                                               \
    static std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;                                 \
    static std::list<CacheTuple> hash_cache_;                                                                       \
    static const size_t capacity_{1024};                                                                            \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                                    \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);       \
    auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);                                    \
    auto return_values = GET_EXECUTOR_FOR_PYBOOST(aclnn_name, __VA_ARGS__);                                         \
    auto ws_size = std::get<0>(return_values);                                                                      \
    auto executor_handle = std::get<1>(return_values);                                                              \
    auto release_function = std::get<3>(return_values);                                                             \
    auto update_function = std::get<4>(return_values);                                                              \
    if (ws_size == 0) {                                                                                             \
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, nullptr, 0, executor_handle, stream_ptr, release_function, \
                             update_function);                                                                      \
    } else {                                                                                                        \
      auto work_ptr = std::make_shared<MemBlock>(device_context, ws_size, stream_id);                               \
      DISPATCH_LAUNCH_KERNEL(device_context, aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr,      \
                             release_function, update_function);                                                    \
    }                                                                                                               \
    static auto sync = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE) ||             \
                       MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON";              \
    if (sync) {                                                                                                     \
      if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                       \
        MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                             \
      }                                                                                                             \
    } else {                                                                                                        \
      runtime::DeviceAddressUtils::ProcessCrossStreamAddress(aclnn_name, device_context, stream_id, __VA_ARGS__);   \
    }                                                                                                               \
  } while (false)

#define LAUNCH_KERNEL(aclnn_name, ws_ptr, ws_size, executor, stream, update_func)                                     \
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeLaunchTask, \
                                     aclnn_name, false);                                                              \
  if (update_func != nullptr) {                                                                                       \
    update_func();                                                                                                    \
  }                                                                                                                   \
  MS_LOG(DEBUG) << "launch task start, " << aclnn_name;                                                               \
  RUN_OP_API_SYNC(aclnn_name, ws_ptr, ws_size, executor, stream);                                                     \
  MS_LOG(DEBUG) << "launch task end, " << aclnn_name;

#define LAUNCH_ACLNN_SYNC(aclnn_api, device_context, stream_id, ...)                                          \
  [](const std::string &aclnn_name, const device::DeviceContext *device_context, size_t real_stream_id,       \
     auto &... args) -> auto {                                                                                \
    static std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;                           \
    static std::list<CacheTuple> hash_cache_;                                                                 \
    static const size_t capacity_{1024};                                                                      \
    runtime::Pipeline::Get().WaitForward();                                                                   \
    runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,                              \
                                             runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false); \
    auto stream_ptr = device_context->device_res_manager_->GetStream(real_stream_id);                         \
    auto return_values = GET_EXECUTOR_FOR_PYBOOST(aclnn_name, args...);                                       \
    auto ws_size = std::get<0>(return_values);                                                                \
    auto executor_handle = std::get<1>(return_values);                                                        \
    auto update_function = std::get<4>(return_values);                                                        \
    if (ws_size == 0) {                                                                                       \
      LAUNCH_KERNEL(aclnn_name, nullptr, 0, executor_handle, stream_ptr, update_function);                    \
    } else {                                                                                                  \
      auto work_ptr = std::make_shared<MemBlock>(device_context, ws_size, real_stream_id);                    \
      LAUNCH_KERNEL(aclnn_name, work_ptr->ptr_, ws_size, executor_handle, stream_ptr, update_function);       \
    }                                                                                                         \
    if (!device::ascend::AscendStreamMng::GetInstance().SyncAllStreams()) {                                   \
      MS_LOG(EXCEPTION) << "SyncStream failed for op " << aclnn_name;                                         \
    }                                                                                                         \
    return return_values;                                                                                     \
  }                                                                                                           \
  (#aclnn_api, device_context, stream_id, __VA_ARGS__)

namespace mindspore {
namespace kernel {
namespace pyboost {
struct MemBlock {
  MemBlock(const DeviceContext *device_context, size_t size, uint32_t stream_id) {
    ptr_ = device_context->device_res_manager_->AllocateMemory(size, stream_id);
    if (ptr_ == nullptr) {
      MS_LOG(EXCEPTION) << "Alloc failed, size:" << size << ", stream_id:" << stream_id;
    }
    device_context_ = device_context;
  }
  ~MemBlock() { device_context_->device_res_manager_->FreeMemory(ptr_); }
  void *ptr_;
  const DeviceContext *device_context_;
};
using MemBlockPtr = std::shared_ptr<MemBlock>;
int8_t GetCubeMathType();
std::pair<int64_t, int64_t> UpdateGeneratorState(const tensor::BaseTensorPtr &seed, const tensor::BaseTensorPtr &offset,
                                                 int64_t step = 10);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_ACLNN_UTILS_H_
