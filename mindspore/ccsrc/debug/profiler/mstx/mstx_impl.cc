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
#include "debug/profiler/mstx/mstx_impl.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include <mutex>
#include "debug/profiler/profiling.h"
#include "debug/profiler/utils.h"

namespace mindspore {
namespace profiler {
namespace {
bool IsMsptiEnableImpl() {
  bool ret = false;
  const char *envVal = std::getenv("LD_PRELOAD");
  if (envVal == nullptr) {
    return ret;
  }
  static const std::string soName = "libmspti.so";
  std::stringstream ss(envVal);
  std::string path;
  while (std::getline(ss, path, ':')) {
    path = mindspore::profiler::Utils::RealPath(path);
    if ((path.size() > soName.size()) && (path.substr(path.size() - soName.size()) == soName)) {
      ret = true;
      break;
    }
  }
  return ret;
}

void InitMstxApis() {
  const char *envVal = std::getenv("ASCEND_HOME_PATH");
  if (envVal == nullptr) {
    return;
  }
  std::string ascendPath = std::string(envVal);
  ascendPath = mindspore::profiler::Utils::RealPath(ascendPath);
  if (ascendPath.empty()) {
    return;
  }
  LoadMstxApiSymbol(ascendPath);
}

}  // namespace

static std::mutex g_mstxRangeIdsMtx;
static std::unordered_map<uint64_t, uint64_t> g_mstxInfoTime;

MstxImpl::MstxImpl() {
  InitMstxApis();
  isMstxSupport_ = IsCannSupportMstxApi();
  isMstxDomainSupport_ = IsCannSupportMstxDomainApi();
}

void MstxImpl::ProfEnable() {
  MS_LOG(INFO) << "enable mstx";
  isProfEnable_.store(true);
}

void MstxImpl::ProfDisable() {
  MS_LOG(INFO) << "disable mstx";
  isProfEnable_.store(false);
}

bool MstxImpl::IsMsptiEnable() {
  static bool isEnable = IsMsptiEnableImpl();
  return isEnable;
}

bool MstxImpl::IsEnable() { return isProfEnable_.load() || IsMsptiEnable(); }

bool MstxImpl::IsSupportMstxApi(bool withDomain) { return withDomain ? isMstxDomainSupport_ : isMstxSupport_; }

void MstxImpl::MarkAImpl(mstxDomainHandle_t domain, const char *message, void *stream) {
  if (!IsEnable()) {
    return;
  }
  if (!IsSupportMstxApi(domain != nullptr)) {
    return;
  }
  uint64_t startTime = mindspore::profiler::Utils::getClockSyscnt();
  if (domain) {
    CALL_MSTX_API(mstxDomainMarkA, domain, message, stream);
  } else {
    CALL_MSTX_API(mstxMarkA, message, stream);
  }
  uint64_t endTime = mindspore::profiler::Utils::getClockSyscnt();
  mindspore::profiler::CollectHostInfo(MSTX_MODULE, MSTX_EVENT, MSTX_STAGE_MARK, startTime, endTime);
}

uint64_t MstxImpl::RangeStartAImpl(mstxDomainHandle_t domain, const char *message, void *stream) {
  uint64_t txTaskId = 0;
  if (!IsEnable()) {
    return txTaskId;
  }
  if (!IsSupportMstxApi(domain != nullptr)) {
    return txTaskId;
  }
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  if (domain) {
    txTaskId = CALL_MSTX_API(mstxDomainRangeStartA, domain, message, stream);
  } else {
    txTaskId = CALL_MSTX_API(mstxRangeStartA, message, stream);
  }
  if (txTaskId == 0) {
    MS_LOG(WARNING) << "Failed to call mstx range startA func for message " << message;
  }
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    g_mstxInfoTime.emplace(txTaskId, startTime);
  }
  return txTaskId;
}

void MstxImpl::RangeEndImpl(mstxDomainHandle_t domain, uint64_t txTaskId) {
  if (!IsEnable()) {
    return;
  }
  if (!IsSupportMstxApi(domain != nullptr)) {
    return;
  }
  uint64_t startTime = 0;
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    auto iter = g_mstxInfoTime.find(txTaskId);
    if (iter == g_mstxInfoTime.end()) {
      MS_LOG(WARNING) << "Failed to find range start time for input range end id " << txTaskId;
      return;
    }
    startTime = iter->second;
    g_mstxInfoTime.erase(iter);
  }
  if (domain) {
    CALL_MSTX_API(mstxDomainRangeEnd, domain, txTaskId);
  } else {
    CALL_MSTX_API(mstxRangeEnd, txTaskId);
  }
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo(MSTX_MODULE, MSTX_EVENT, MSTX_STAGE_RANGE, startTime, endTime);
}

mstxDomainHandle_t MstxImpl::DomainCreateAImpl(const char *domainName) {
  if (!IsSupportMstxApi(true)) {
    return nullptr;
  }
  std::string nameStr = std::string(domainName);
  std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
  auto iter = domains_.find(nameStr);
  if (iter != domains_.end()) {
    return iter->second;
  }
  mstxDomainHandle_t handle = CALL_MSTX_API(mstxDomainCreateA, domainName);
  if (handle != nullptr) {
    domains_.emplace(nameStr, handle);
  }
  return handle;
}

void MstxImpl::DomainDestroyImpl(mstxDomainHandle_t domain) {
  if (!IsSupportMstxApi(true)) {
    return;
  }
  if (domain == nullptr) {
    return;
  }
  CALL_MSTX_API(mstxDomainDestroy, domain);
}

}  // namespace profiler
}  // namespace mindspore
