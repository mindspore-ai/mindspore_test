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
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>
#include "debug/profiler/profiling.h"
#include "debug/profiler/profiler.h"
#include "pynative/pynative_utils.h"
#include "pynative/forward/forward_task.h"
#include "plugin/device/ascend/hal/profiler/mstx/mstx_mgr.h"
#include "plugin/device/ascend/hal/profiler/mstx/mstx_symbol.h"
#include "plugin/res_manager/ascend/hccl_adapter/plugin/hccl_plugin.h"
#include "runtime/pynative/task/device_task.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"

namespace mindspore {
namespace profiler {
namespace ascend {

static std::mutex g_mstxRangeIdsMtx;
static std::unordered_map<uint64_t, uint64_t> g_mstxRangeIds;
static std::unordered_map<uint64_t, uint64_t> g_mstxInfoTime;
static std::unordered_map<MstxTaskType, std::string> reportStageNames = {
  {MstxTaskType::mark, "Mark"},
  {MstxTaskType::start, "RangeStart"},
  {MstxTaskType::end, "RangeEnd"},
};

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

static void SetStreamForCurrentThread() {
  // set current context for each pipeline thread, so that stream can be used to launch tx task
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ascend::AscendHalManager::GetInstance().SetContext(device_id);
}

static void DispatchFrontendTask(const std::shared_ptr<runtime::AsyncTask> &task) {
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    MS_LOG(INFO) << "PyBoost sync run frontend task";
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(task->task_id());
    runtime::Pipeline::Get().frontend_stage()->Push(task);
  }
}

static void DispatchDeviceTask(std::function<void(void)> run_func, MstxTaskType type) {
  auto deviceTask = std::make_shared<MstxDeviceTask>(run_func, type);
  MS_EXCEPTION_IF_NULL(deviceTask);
  runtime::ProfilerAnalyzer::GetInstance().RecordFlowData(deviceTask->task_id());
  runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(deviceTask);
}

static void DispatchMarkTask(const char *message, void *stream) {
  // in case input msg is released before use it, create new message
  auto msgPtr = std::make_shared<std::string>(message);
  MS_EXCEPTION_IF_NULL(msgPtr);

  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [msgPtr, stream]() {
      auto txTask = [msgPtr, stream]() {
        runtime::OpExecutor::DispatchLaunchTask([msgPtr, stream]() {
          SetStreamForCurrentThread();
          MstxMgr::MarkImpl(msgPtr->c_str(), stream);
        });
      };
      if (!runtime::OpExecutor::NeedSync()) {
        DispatchDeviceTask(txTask, MstxTaskType::mark);
      } else {
        txTask();
      }
    },
    MstxTaskType::mark));
}

static void DispatchRangeStartTask(const char *message, void *stream, uint64_t msRangeId) {
  // in case input msg is released before use it, create new message
  auto msgPtr = std::make_shared<std::string>(message);
  MS_EXCEPTION_IF_NULL(msgPtr);

  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [msgPtr, stream, msRangeId]() {
      auto txTask = [msgPtr, stream, msRangeId]() {
        runtime::OpExecutor::DispatchLaunchTask([msgPtr, stream, msRangeId]() {
          SetStreamForCurrentThread();
          MstxMgr::RangeStartImpl(msgPtr->c_str(), stream, msRangeId);
        });
      };
      if (!runtime::OpExecutor::NeedSync()) {
        DispatchDeviceTask(txTask, MstxTaskType::start);
      } else {
        txTask();
      }
    },
    MstxTaskType::start));
}

static void DispatchRangeEndTask(uint64_t msRangeId) {
  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [msRangeId]() {
      auto txTask = [msRangeId]() {
        runtime::OpExecutor::DispatchLaunchTask([msRangeId]() {
          SetStreamForCurrentThread();
          MstxMgr::RangeEndImpl(msRangeId);
        });
      };
      if (!runtime::OpExecutor::NeedSync()) {
        DispatchDeviceTask(txTask, MstxTaskType::end);
      } else {
        txTask();
      }
    },
    MstxTaskType::end));
}

void MstxFrontendTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute,
                                     reportStageNames[type_], false, false, task_id_);
  run_func_();
}

void MstxDeviceTask::Run() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute,
                                     reportStageNames[type_], false, false, task_id_);
  run_func_();
}

MstxMgr::MstxMgr() {
  std::string ascend_path = mindspore::device::ascend::GetAscendPath();
  LoadMstxApiSymbol(ascend_path);
}

void MstxMgr::MarkImpl(const char *message, void *stream) {
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  CALL_MSTX_API(mstxMarkA, message, stream);
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo("Ascend", "Mstx", "Mark", startTime, endTime);
}

void MstxMgr::RangeStartImpl(const char *message, void *stream, uint64_t msRangeId) {
  uint64_t startTime = mindspore::profiler::GetClockSyscnt();
  uint64_t taskId = CALL_MSTX_API(mstxRangeStartA, message, stream);
  if (taskId == 0) {
    MS_LOG(WARNING) << "Failed to call mstxRangeStartA func.";
    return;
  }
  std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
  g_mstxRangeIds.insert(std::make_pair(msRangeId, taskId));
  g_mstxInfoTime.insert(std::make_pair(msRangeId, startTime));
}

void MstxMgr::RangeEndImpl(uint64_t msRangeId) {
  uint64_t taskId = 0;
  uint64_t startTime = 0;
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    auto iter = g_mstxRangeIds.find(msRangeId);
    if (iter == g_mstxRangeIds.end()) {
      MS_LOG(WARNING) << "Failed to find range start id for input range end id " << msRangeId;
      return;
    }
    taskId = iter->second;
    g_mstxRangeIds.erase(iter);
    iter = g_mstxInfoTime.find(msRangeId);
    if (iter == g_mstxInfoTime.end()) {
      MS_LOG(WARNING) << "Failed to find range start time for input range end id " << msRangeId;
      return;
    }
    startTime = iter->second;
    g_mstxInfoTime.erase(iter);
  }
  CALL_MSTX_API(mstxRangeEnd, taskId);
  uint64_t endTime = mindspore::profiler::GetClockSyscnt();
  mindspore::profiler::CollectHostInfo("Ascend", "Mstx", "Range", startTime, endTime);
}

void MstxMgr::Mark(const char *message, void *stream) {
  MS_LOG(INFO) << "Start to mstx mark";
  if (!IsEnable()) {
    return;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute, "Mark", false,
                                     true);
  if (stream == nullptr) {
    MarkImpl(message, stream);
  } else {
    DispatchMarkTask(message, stream);
  }
}

uint64_t MstxMgr::RangeStart(const char *message, void *stream) {
  MS_LOG(INFO) << "Start to mstx range start";
  if (!IsEnable()) {
    return 0;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute, "RangeStart",
                                     false, true);
  uint64_t id = msRangeId_++;
  if (stream == nullptr) {
    RangeStartImpl(message, stream, id);
  } else {
    {
      std::lock_guard<std::mutex> lock(idStreamsMtx_);
      msRangeIdsWithStream_.insert(id);
    }
    DispatchRangeStartTask(message, stream, id);
  }
  return id;
}

void MstxMgr::RangeEnd(uint64_t msRangeId) {
  MS_LOG(INFO) << "Start to mstx range end, id: " << msRangeId;
  if (msRangeId == 0 || !IsEnable()) {
    return;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute, "RangeEnd",
                                     false, true);
  bool rangeIdWithStream = false;
  {
    std::lock_guard<std::mutex> lock(idStreamsMtx_);
    if (msRangeIdsWithStream_.find(msRangeId) != msRangeIdsWithStream_.end()) {
      rangeIdWithStream = true;
      msRangeIdsWithStream_.erase(msRangeId);
    }
  }
  if (!rangeIdWithStream) {
    RangeEndImpl(msRangeId);
  } else {
    DispatchRangeEndTask(msRangeId);
  }
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
