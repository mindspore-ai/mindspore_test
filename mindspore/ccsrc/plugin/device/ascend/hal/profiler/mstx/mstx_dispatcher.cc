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
#include "debug/profiler/profiler.h"
#include "pynative/pynative_utils.h"
#include "pynative/forward/forward_task.h"
#include "plugin/device/ascend/hal/profiler/mstx/mstx_dispatcher.h"
#include "plugin/res_manager/ascend/hccl_adapter/plugin/hccl_plugin.h"
#include "runtime/pynative/task/device_task.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"

namespace mindspore {
namespace profiler {
namespace ascend {

static std::mutex g_mstxRangeIdsMtx;
static std::unordered_map<uint64_t, uint64_t> g_mstxRangeIds;
static std::unordered_map<MstxTaskType, std::string> reportStageNames = {
  {MstxTaskType::mark, "Mark"},
  {MstxTaskType::start, "RangeStart"},
  {MstxTaskType::end, "RangeEnd"},
};

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

void MstxDispatcher::RangeStartImpl(mstxDomainHandle_t domain, const char *message, void *stream, uint64_t msRangeId) {
  uint64_t taskId = MstxImpl::GetInstance().RangeStartAImpl(domain, message, stream);
  if (taskId == 0) {
    MS_LOG(WARNING) << "Failed to call mstx range start func.";
    return;
  }
  std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
  g_mstxRangeIds.insert(std::make_pair(msRangeId, taskId));
}

void MstxDispatcher::RangeEndImpl(mstxDomainHandle_t domain, uint64_t msRangeId) {
  uint64_t taskId = 0;
  {
    std::lock_guard<std::mutex> lock(g_mstxRangeIdsMtx);
    auto iter = g_mstxRangeIds.find(msRangeId);
    if (iter == g_mstxRangeIds.end()) {
      MS_LOG(WARNING) << "Failed to find range start id for input range end id " << msRangeId;
      return;
    }
    taskId = iter->second;
    g_mstxRangeIds.erase(iter);
  }
  MstxImpl::GetInstance().RangeEndImpl(domain, taskId);
}

void MstxDispatcher::DispatchMarkTask(mstxDomainHandle_t domain, const char *message, void *stream) {
  // in case input msg is released before use it, create new message
  auto msgPtr = std::make_shared<std::string>(message);
  MS_EXCEPTION_IF_NULL(msgPtr);

  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [domain, msgPtr, stream]() {
      auto txTask = [domain, msgPtr, stream]() {
        runtime::OpExecutor::DispatchLaunchTask([domain, msgPtr, stream]() {
          SetStreamForCurrentThread();
          MstxImpl::GetInstance().MarkAImpl(domain, msgPtr->c_str(), stream);
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

void MstxDispatcher::DispatchRangeStartTask(mstxDomainHandle_t domain, const char *message, void *stream,
                                            uint64_t msRangeId) {
  // in case input msg is released before use it, create new message
  auto msgPtr = std::make_shared<std::string>(message);
  MS_EXCEPTION_IF_NULL(msgPtr);

  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [domain, msgPtr, stream, msRangeId]() {
      auto txTask = [domain, msgPtr, stream, msRangeId]() {
        runtime::OpExecutor::DispatchLaunchTask([domain, msgPtr, stream, msRangeId]() {
          SetStreamForCurrentThread();
          RangeStartImpl(domain, msgPtr->c_str(), stream, msRangeId);
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

void MstxDispatcher::DispatchRangeEndTask(mstxDomainHandle_t domain, uint64_t msRangeId) {
  DispatchFrontendTask(std::make_shared<MstxFrontendTask>(
    [domain, msRangeId]() {
      auto txTask = [domain, msRangeId]() {
        runtime::OpExecutor::DispatchLaunchTask([domain, msRangeId]() {
          SetStreamForCurrentThread();
          MstxDispatcher::RangeEndImpl(domain, msRangeId);
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

void MstxDispatcher::Mark(const char *message, void *stream, mstxDomainHandle_t domain) {
  MS_LOG(INFO) << "Start to run mstx mark for message: " << message;
  if (!IsEnable()) {
    return;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute,
                                     MSTX_OP_NAME_MARK, false, true);
  if (stream == nullptr) {
    MstxImpl::GetInstance().MarkAImpl(domain, message, stream);
  } else {
    DispatchMarkTask(domain, message, stream);
  }
}

uint64_t MstxDispatcher::RangeStart(const char *message, void *stream, mstxDomainHandle_t domain) {
  MS_LOG(INFO) << "Start to run mstx range start for message: " << message;
  if (!IsEnable()) {
    return 0;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute,
                                     MSTX_OP_NAME_RANGE_START, false, true);
  uint64_t id = msRangeId_++;
  if (stream == nullptr) {
    RangeStartImpl(domain, message, stream, id);
  } else {
    {
      std::lock_guard<std::mutex> lock(idStreamsMtx_);
      msRangeIdsWithStream_.insert(id);
    }
    DispatchRangeStartTask(domain, message, stream, id);
  }
  return id;
}

void MstxDispatcher::RangeEnd(uint64_t msRangeId, mstxDomainHandle_t domain) {
  MS_LOG(INFO) << "Start to run mstx range end, id: " << msRangeId;
  if (msRangeId == 0 || !IsEnable()) {
    return;
  }
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kOther, runtime::ProfilerEvent::kExecute,
                                     MSTX_OP_NAME_RANGE_END, false, true);
  bool rangeIdWithStream = false;
  {
    std::lock_guard<std::mutex> lock(idStreamsMtx_);
    if (msRangeIdsWithStream_.find(msRangeId) != msRangeIdsWithStream_.end()) {
      rangeIdWithStream = true;
      msRangeIdsWithStream_.erase(msRangeId);
    }
  }
  if (!rangeIdWithStream) {
    RangeEndImpl(domain, msRangeId);
  } else {
    DispatchRangeEndTask(domain, msRangeId);
  }
}

mstxDomainHandle_t MstxDispatcher::DomainCreate(const char *name) {
  return MstxImpl::GetInstance().DomainCreateAImpl(name);
}

void MstxDispatcher::DomainDestroy(mstxDomainHandle_t domain) { MstxImpl::GetInstance().DomainDestroyImpl(domain); }

void MstxDispatcher::Enable() {
  MS_LOG(INFO) << "enable mstx";
  MstxImpl::GetInstance().ProfEnable();
}

void MstxDispatcher::Disable() {
  MS_LOG(INFO) << "disable mstx";
  MstxImpl::GetInstance().ProfDisable();
}

bool MstxDispatcher::IsEnable() { return MstxImpl::GetInstance().IsEnable(); }

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
