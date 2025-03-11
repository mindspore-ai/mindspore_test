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

#ifndef MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_
#define MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include "include/common/visible.h"
#include "debug/profiler/mstx/mstx_symbol.h"

namespace mindspore {
namespace profiler {

const char MSTX_MODULE[] = "Ascend";
const char MSTX_EVENT[] = "Mstx";
const char MSTX_STAGE_MARK[] = "Mark";
const char MSTX_STAGE_RANGE[] = "Range";

const char MSTX_DOMAIN_COMMUNICATION[] = "communication";
const char MSTX_DOMAIN_DEFAULT[] = "default";
const char MSTX_GETNEXT[] = "GetNext";

class PROFILER_EXPORT MstxImpl {
 public:
  MstxImpl();
  ~MstxImpl() = default;

  static MstxImpl &GetInstance() {
    static MstxImpl instance;
    return instance;
  }

  void MarkAImpl(mstxDomainHandle_t domain, const char *message, void *stream);
  uint64_t RangeStartAImpl(mstxDomainHandle_t domain, const char *message, void *stream);
  void RangeEndImpl(mstxDomainHandle_t domain, uint64_t txTaskId);
  mstxDomainHandle_t DomainCreateAImpl(const char *domainName);
  void DomainDestroyImpl(mstxDomainHandle_t domain);

  void ProfEnable();
  void ProfDisable();
  bool IsEnable();

 private:
  bool IsMsptiEnable();
  bool IsSupportMstxApi(bool withDomain);

 private:
  std::atomic<bool> isProfEnable_{false};
  bool isMstxSupport_{false};
  bool isMstxDomainSupport_{false};
  std::mutex domainMtx_;
  std::unordered_map<std::string, mstxDomainHandle_t> domains_;
};

#define MSTX_START(rangeId, message, stream, domainName)                                                   \
  do {                                                                                                     \
    auto domainHandle = mindspore::profiler::MstxImpl::GetInstance().DomainCreateAImpl(domainName);        \
    rangeId = mindspore::profiler::MstxImpl::GetInstance().RangeStartAImpl(domainHandle, message, stream); \
  } while (0);

#define MSTX_END(rangeId, domainName)                                                               \
  do {                                                                                              \
    auto domainHandle = mindspore::profiler::MstxImpl::GetInstance().DomainCreateAImpl(domainName); \
    mindspore::profiler::MstxImpl::GetInstance().RangeEndImpl(domainHandle, rangeId);               \
  } while (0);

#define MSTX_START_WITHOUT_DOMAIN(rangeId, message, stream)                                           \
  do {                                                                                                \
    rangeId = mindspore::profiler::MstxImpl::GetInstance().RangeStartAImpl(nullptr, message, stream); \
  } while (0);

#define MSTX_END_WITHOUT_DOMAIN(rangeId)                                         \
  do {                                                                           \
    mindspore::profiler::MstxImpl::GetInstance().RangeEndImpl(nullptr, rangeId); \
  } while (0);

}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXIMPL_H_
