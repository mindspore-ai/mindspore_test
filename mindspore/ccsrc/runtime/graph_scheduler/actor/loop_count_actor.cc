/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/loop_count_actor.h"
#include <set>
#include "runtime/graph_scheduler/actor/data_prepare_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "runtime/graph_scheduler/actor/profiler_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"
#include "runtime/graph_scheduler/execution_order_check/comm_execution_order_check.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "runtime/device/stream_synchronizer.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "runtime/graph_scheduler/rpc_node_scheduler.h"
#endif

namespace mindspore {
namespace runtime {
using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

void LoopCountActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Need wait MemoryManagerActor running finished to avoid the illegal memory timing problem before
  // LoopCountActor exits, because other processors which are not in actor also will process device tensor.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
}

void LoopCountActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  IncreaseLoopCount(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  total_running_count_++;
  current_count_++;
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run graph failed and please check error log.";
    return;
  }

  static auto &process = Process::GetInstance();
  process.CheckCommOrderIteration(total_running_count_);

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
  }

  if (profiler_aid_ != nullptr) {
    MS_LOG(INFO) << "Sync stream in the step end by profiler.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    SendProfilerReq(context);
  }

  if (first_control_aids_.empty() && entrance_aids_.empty()) {
    RealRun(context);
    return;
  }
  HandleNotifyOnePhase(context);
}

void LoopCountActor::RealRun(OpContext<DeviceTensor> *const context) {
  notify_messages_.clear();
  // Sync device stream.
  if ((strategy_ == GraphExecutionStrategy::kPipeline) && is_need_sync_stream_) {
    MS_LOG(INFO) << "Sync stream in the step end.";
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kStreamSync, GetAID().Name());
    std::set<const DeviceContext *> sync_stream_device_contexts;
    for (auto &device_context : device_contexts_) {
      MS_EXCEPTION_IF_NULL(device_context);
      if ((sync_stream_device_contexts.count(device_context) == 0) &&
          (!device::StreamSynchronizer::GetInstance()->SyncStream(device_context->device_context_key().device_name_))) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                          ("Sync stream failed:" + device_context->device_context_key().ToString()));
      }
      (void)sync_stream_device_contexts.insert(device_context);

      // Trigger disaster recovery and exit loop early.
      if (RecoveryContext::GetInstance()->enable_recovery() && CollectiveManager::instance()->need_reinit()) {
        current_count_ = loop_count_;
      }
    }
    MS_LOG(INFO) << "Sync stream success.";
  }

  PostRun(context);
}

void LoopCountActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID(), total_running_count_,
                            sink_size_);
}

void LoopCountActor::SendProfilerReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::SendSync(*profiler_aid_, &ProfilerActor::ProfilerOnStepEnd, context, &GetAID(),
                            total_running_count_);
}

void LoopCountActor::HandleNotifyOnePhase(OpContext<DeviceTensor> *const context) {
  if (first_control_aids_.empty()) {
    HandleNotifyTwoPhase(context);
    return;
  }
  for (auto &first_control_aid : first_control_aids_) {
    ActorDispatcher::Send(first_control_aid, &AbstractActor::HandleWaitMessage, context, GetAID());
  }
}

void LoopCountActor::HandleNotifyTwoPhase(OpContext<DeviceTensor> *const context) {
  if (entrance_aids_.empty()) {
    RealRun(context);
    return;
  }
  // Send to EntranceActor to clear the data which are generated in the loop body execution.
  for (auto &entrance_aid : entrance_aids_) {
    ActorDispatcher::Send(entrance_aid, &AbstractActor::HandleWaitMessage, context, GetAID());
  }
  return;
}

void LoopCountActor::HandleNotifyMessage(OpContext<DeviceTensor> *const context, const AID &from_aid) {
  notify_messages_.emplace_back(from_aid);
  MS_LOG(DEBUG) << "Actor:" << GetAID() << " receive signal message from actor:" << from_aid
                << " current size:" << notify_messages_.size() << " need size:" << first_control_aids_.size() << " and"
                << entrance_aids_.size() << " for actor:" << GetAID();
  if (notify_messages_.size() < first_control_aids_.size() ||
      (notify_messages_.size() < first_control_aids_.size() + entrance_aids_.size() &&
       notify_messages_.size() > first_control_aids_.size())) {
    return;
  }

  if (notify_messages_.size() == first_control_aids_.size()) {
    MS_LOG(DEBUG) << "Handle first control aid finish for actor:" << GetAID();
    HandleNotifyTwoPhase(context);
    return;
  }

  if (notify_messages_.size() == first_control_aids_.size() + entrance_aids_.size()) {
    MS_LOG(DEBUG) << "Handle first control aid and entrance aid finish for actor:" << GetAID();
    RealRun(context);
    return;
  }

  std::stringstream ofs;
  ofs << "Invalid input signals size:" << notify_messages_.size()
      << " first control aid size:" << first_control_aids_.size() << " entrance aid size:" << entrance_aids_.size()
      << " for actor:" << GetAID();
  SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), ofs.str());
}

void LoopCountActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }

  // Send output control.
  auto from_aid = const_cast<AID *>(&GetAID());
  for (auto &output_control : output_control_arrows_) {
    MS_EXCEPTION_IF_NULL(output_control);
    ActorDispatcher::Send(output_control->to_op_id_, &OpActor::RunOpControl, from_aid, context);
  }

#if defined(__linux__) && defined(WITH_BACKEND)
  // Flush sent data after each step is done.
  RpcActorStatusUpdater::GetInstance().FlushRpcData(graph_name_);
#endif

  // The LoopCountActor exits.
  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send to DataPrepareActor to trigger next step running.
  ActorDispatcher::Send(data_prepare_aid_, &OpActor::RunOpControl, from_aid, context);
}
}  // namespace runtime
}  // namespace mindspore
