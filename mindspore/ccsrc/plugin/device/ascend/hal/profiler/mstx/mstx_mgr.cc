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

#include <map>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>
#include "plugin/device/ascend/hal/profiler/mstx/mstx_mgr.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/hal/profiler/mstx/mstx_symbol.h"
#include "plugin/device/ascend/hal/hccl_adapter/plugin/hccl_plugin.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace profiler {
namespace ascend {

static std::mutex g_mstxInfoTimeMtx;
static std::unordered_map<uint64_t, uint64_t> g_mstxInfoTime;

static std::string RealPath(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return "";
  }
  char realPath[PATH_MAX] = {0};
  if (realpath(path.c_str(), realPath) == nullptr) {
    return "";
  }
  return std::string(realPath);
}

MstxMgr::MstxMgr() {
  std::string ascend_path = mindspore::transform::GetAscendPath();
  LoadMstxApiSymbol(ascend_path);
}

void MstxMgr::Mark(const char *message, void *stream) {
  MS_LOG(INFO) << "Start to mstx mark";
  if (!IsEnable()) {
    return;
  }
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  CALL_MSTX_API(mstxMarkA, message, stream);
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo("Ascend", "mstx", "mark", startTime, endTime);
}

uint64_t MstxMgr::RangeStart(const char *message, void *stream) {
  MS_LOG(INFO) << "Start to mstx range start";
  if (!IsEnable()) {
    return 0;
  }
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  auto id = CALL_MSTX_API(mstxRangeStartA, message, stream);
  if (id == 0) {
    return 0;
  } else {
    {
      std::lock_guard<std::mutex> lock(g_mstxInfoTimeMtx);
      g_mstxInfoTime.insert(std::make_pair(id, startTime));
    }
    return id;
  }
}

void MstxMgr::RangeEnd(uint64_t msRangeId) {
  MS_LOG(INFO) << "Start to mstx range end, id: " << msRangeId;
  if (msRangeId == 0 || !IsEnable()) {
    return;
  }
  uint64_t startTime = 0;
  {
    std::lock_guard<std::mutex> lock(g_mstxInfoTimeMtx);
    auto timeIter = g_mstxInfoTime.find(msRangeId);
    if (timeIter == g_mstxInfoTime.end()) {
      return;
    }
    startTime = timeIter->second;
    g_mstxInfoTime.erase(timeIter);
  }
  CALL_MSTX_API(mstxRangeEnd, msRangeId);
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo("Ascend", "mstx", "range", startTime, endTime);
}

void MstxMgr::Enable() {
  MS_LOG(INFO) << "enable mstx";
  isEnable_.store(true);
}

void MstxMgr::Disable() {
  MS_LOG(INFO) << "disable mstx";
  isEnable_.store(false);
}

bool MstxMgr::IsProfEnable() { return isEnable_.load(); }

bool MstxMgr::IsMsptiEnableImpl() {
  bool ret = false;
  const char *envVal = std::getenv("LD_PRELOAD");
  if (envVal == nullptr) {
    return ret;
  }
  static const std::string soName = "libmspti.so";
  std::stringstream ss(envVal);
  std::string path;
  while (std::getline(ss, path, ':')) {
    path = RealPath(path);
    if ((path.size() > soName.size()) && (path.substr(path.size() - soName.size()) == soName)) {
      ret = true;
      break;
    }
  }
  return ret;
}

bool MstxMgr::IsMsptiEnable() {
  static bool isEnable = IsMsptiEnableImpl();
  return isEnable;
}

bool MstxMgr::IsEnable() { return IsProfEnable() || IsMsptiEnable(); }

std::string GetMstxHcomMsg(const std::string &opName, size_t dataCnt, HcclDataType dataType, HcclComm comm) {
  static const std::map<HcclDataType, std::string> dataTypes = {
    {HCCL_DATA_TYPE_INT8, "int8"},     {HCCL_DATA_TYPE_INT16, "int16"}, {HCCL_DATA_TYPE_INT32, "int32"},
    {HCCL_DATA_TYPE_FP16, "fp16"},     {HCCL_DATA_TYPE_FP32, "fp32"},   {HCCL_DATA_TYPE_INT64, "int64"},
    {HCCL_DATA_TYPE_UINT64, "uint64"}, {HCCL_DATA_TYPE_UINT8, "uint8"}, {HCCL_DATA_TYPE_UINT16, "uint16"},
    {HCCL_DATA_TYPE_UINT32, "uint32"}, {HCCL_DATA_TYPE_FP64, "fp64"},   {HCCL_DATA_TYPE_BFP16, "bfp16"}};
  constexpr int64_t MAX_GROUP_NAME_LEN = 128;
  static char commName[MAX_GROUP_NAME_LEN] = {0};
  static std::map<HcclComm, std::string> commNames;
  if (!MstxMgr::GetInstance().IsEnable()) {
    return "";
  }
  std::vector<std::string> opDescVec;
  opDescVec.push_back(opName);
  auto nameIter = commNames.find(comm);
  if (nameIter != commNames.end()) {
    opDescVec.push_back(nameIter->second);
  } else {
    if (HcclGetCommName(comm, commName) != HCCL_SUCCESS) {
      opDescVec.push_back("na");
    } else {
      std::string name(commName);
      opDescVec.push_back(name);
      commNames.insert({comm, name});
    }
  }
  auto iter = dataTypes.find(dataType);
  if (iter != dataTypes.end()) {
    opDescVec.push_back(iter->second);
  } else {
    opDescVec.push_back("na");
  }
  opDescVec.push_back(std::to_string(dataCnt));
  std::string opDescMsg =
    std::accumulate(opDescVec.begin(), opDescVec.end(), std::string(""),
                    [](const std::string &a, const std::string &b) { return a.empty() ? b : a + "," + b; });
  std::string hcomMsg = "comm:" + opDescMsg;
  return hcomMsg;
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
