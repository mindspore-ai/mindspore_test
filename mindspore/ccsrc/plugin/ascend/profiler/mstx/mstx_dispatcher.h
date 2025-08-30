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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXMGR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXMGR_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include "acl/acl_prof.h"
#include "hccl/hccl_types.h"
#include "runtime/pipeline/task/task.h"
#include "tools/profiler/mstx/mstx_impl.h"

namespace mindspore {
namespace profiler {
namespace ascend {

enum class MstxTaskType { mark, start, end };
const char MSTX_OP_NAME_MARK[] = "Mark";
const char MSTX_OP_NAME_RANGE_START[] = "RangeStart";
const char MSTX_OP_NAME_RANGE_END[] = "RangeEnd";

class MstxDispatcher {
 public:
  MstxDispatcher() = default;
  ~MstxDispatcher() = default;

  static MstxDispatcher &GetInstance() {
    static MstxDispatcher instance;
    return instance;
  }

  void Mark(const char *message, void *stream, const std::string &domain_name = "default");
  uint64_t RangeStart(const char *message, void *stream, const std::string &domain_name = "default");
  void RangeEnd(uint64_t id, const std::string &domain_name = "default");

  mstxDomainHandle_t DomainCreate(const char *name);
  void DomainDestroy(mstxDomainHandle_t domain);
  void SetDomain(const std::vector<std::string> &domainInclude, const std::vector<std::string> &domainExclude);

  static void RangeStartImpl(const std::string &domain, const char *message, void *stream, uint64_t msRangeId);
  static void RangeEndImpl(const std::string &domain, uint64_t msRangeId);

  void Enable();
  void Disable();
  bool IsEnable();

  inline bool GetRangeId() { return msRangeId_++; }

 private:
  void DispatchMarkTask(const std::string &domain_name, const char *message, void *stream);
  void DispatchRangeStartTask(const std::string &domain_name, const char *message, void *stream, uint64_t msRangeId);
  void DispatchRangeEndTask(const std::string &domain_name, uint64_t msRangeId);

 private:
  std::atomic<bool> isEnable_{false};
  std::atomic<uint64_t> msRangeId_{1};
  std::mutex idStreamsMtx_;
  std::unordered_set<int> msRangeIdsWithStream_;
};

class MstxFrontendTask : public runtime::AsyncTask {
 public:
  MstxFrontendTask(std::function<void(void)> run_func, MstxTaskType type)
      : runtime::AsyncTask(runtime::kFrontendTask), run_func_(std::move(run_func)), type_(type) {}
  ~MstxFrontendTask() override = default;
  void Run() override;

 private:
  std::function<void(void)> run_func_;
  MstxTaskType type_;
};

class MstxDeviceTask : public runtime::AsyncTask {
 public:
  MstxDeviceTask(std::function<void(void)> run_func, MstxTaskType type)
      : runtime::AsyncTask(runtime::kDeviceOpTask), run_func_(std::move(run_func)), type_(type) {}
  ~MstxDeviceTask() override = default;
  void Run() override;

 private:
  std::function<void(void)> run_func_;
  MstxTaskType type_;
};

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXMGR_H_
