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
#include <unordered_set>
#include <unordered_map>
#include "acl/acl_prof.h"
#include "hccl/hccl_types.h"

namespace mindspore {
namespace profiler {
namespace ascend {
class MstxMgr {
 public:
  MstxMgr();
  ~MstxMgr() = default;

  static MstxMgr &GetInstance() {
    static MstxMgr instance;
    return instance;
  }

  void Mark(const char *message, void *stream);
  uint64_t RangeStart(const char *message, void *stream);
  void RangeEnd(uint64_t id);

  void Enable();
  void Disable();
  bool IsEnable();

 private:
  bool IsProfEnable();
  bool IsMsptiEnable();
  bool IsMsptiEnableImpl();

 private:
  std::atomic<bool> isEnable_{false};
};

struct MstxRange {
  uint64_t rangeId{0};
  MstxRange(const std::string &message, void *stream) {
    if (MstxMgr::GetInstance().IsEnable()) {
      rangeId = MstxMgr::GetInstance().RangeStart(message.c_str(), stream);
    }
  }

  ~MstxRange() {
    if (MstxMgr::GetInstance().IsEnable()) {
      MstxMgr::GetInstance().RangeEnd(rangeId);
    }
  }
};

std::string GetMstxHcomMsg(const std::string &opName, uint64_t dataCnt, HcclDataType dataType, HcclComm comm);
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXMGR_H_
