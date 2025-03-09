/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "utils/ms_context.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#include "debug/debugger/debugger_utils.h"
#include "debug/data_dump/device_statistic/check_overflow.h"
#endif
#include "debug/data_dump/data_dumper.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "debug/profiler/profiling.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "debug/data_dump/overflow_counter.h"
#include "debug/hooker/hook_debugger.h"
#include "debug/hooker/deprecated_env.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace runtime {
void DebugActor::DebugPreLaunch(const AnfNodePtr &node, const std::vector<DeviceTensor *> &input_device_tensors,
                                const std::vector<DeviceTensor *> &output_device_tensors,
                                const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                                const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Load and read data for the given node if needed. Dump the node if dump is enabled and free the loaded
 * memory after the dump (for GPU and ascend kernel-by-kernel).
 */
void DebugActor::DebugPostLaunch(const AnfNodePtr &node, const std::vector<DeviceTensor *> &input_device_tensors,
                                 const std::vector<DeviceTensor *> &output_device_tensors,
                                 const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                                 const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

  if (!node->isa<CNode>()) {
    return;
  }
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "kernel by kernel debug for node: " << cnode->fullname_with_scope() << ".";
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
#ifdef ENABLE_DEBUGGER
    AscendKbkDump(cnode, input_device_tensors, output_device_tensors, device_context);
#endif
  } else if (device_context->GetDeviceType() == device::DeviceType::kCPU) {
    if (DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_LITE_EXCEPTION) {
      MS_LOG(WARNING) << "Abnormal dump is not supported on CPU backend.";
      return;
    }
    if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      MS_EXCEPTION_IF_NULL(kernel_graph);
      CPUE2eDump::DumpCNodeData(cnode, kernel_graph->graph_id());
      CPUE2eDump::DumpRunIter(kernel_graph);
    }
  } else if (device_context->GetDeviceType() == device::DeviceType::kGPU) {
#ifdef ENABLE_DEBUGGER
    if (DumpJsonParser::GetInstance().op_debug_mode() == DumpJsonParser::DUMP_LITE_EXCEPTION) {
      MS_LOG(WARNING) << "Abnormal dump is not supported on GPU backend.";
      return;
    }
    auto debugger = Debugger::GetInstance();
    if (debugger != nullptr) {
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
      debugger->InsertExecutedGraph(kernel_graph);
      bool read_data = CheckReadData(cnode);
      if (read_data) {
        ReadDataAndDump(cnode, input_device_tensors, output_device_tensors, exec_order_, device_context);
      }
    }
    exec_order_ += 1;
#endif
  }
}

/*
 * Feature group: Dump, Ascend.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: Dump data for the given node if needed. It can be normal dump and overflow dump and exception dump
 * (ascend kernel-by-kernel e2e dump).
 */
#ifdef ENABLE_DEBUGGER
void DebugActor::AscendKbkDump(const CNodePtr &cnode, const std::vector<DeviceTensor *> &input_device_tensors,
                               const std::vector<DeviceTensor *> &output_device_tensors,
                               const DeviceContext *device_context) {
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(cnode->func_graph());
    MS_EXCEPTION_IF_NULL(kernel_graph);
    debugger->InsertExecutedGraph(kernel_graph);
    debugger->SetAscendKernelByKernelFlag(true);
    auto &dump_json_parser = DumpJsonParser::GetInstance();
    bool e2e_dump_enabled = dump_json_parser.e2e_dump_enabled();
    uint32_t op_debug_mode = dump_json_parser.op_debug_mode();
    bool abnormal_dump = false;
    bool sync_ok = true;
    bool read_data = false;
    if (!e2e_dump_enabled) {
      exec_order_ += 1;
      return;
    }
    if (op_debug_mode == DumpJsonParser::DUMP_LITE_EXCEPTION) {
      abnormal_dump = true;
      sync_ok = device_ctx_->device_res_manager_->SyncAllStreams();
      if (!sync_ok) {
        MS_LOG(ERROR) << "Sync stream error! The node input will be dumped";
      }
    } else if (op_debug_mode == DumpJsonParser::DUMP_BOTH_OVERFLOW && dump_json_parser.DumpEnabledForIter()) {
      read_data = true;
    } else {
      read_data = CheckReadData(cnode);
    }
    if ((read_data && e2e_dump_enabled) || !sync_ok) {
      if (dump_json_parser.e2e_sync_dump_enabled()) {
        ReadDataAndDump(cnode, input_device_tensors, output_device_tensors, exec_order_, device_context, abnormal_dump);
      } else {
        DumpDataViaCallback(cnode, input_device_tensors, output_device_tensors, device_context);
      }

      if (!sync_ok) {
        MS_LOG(EXCEPTION) << "Sync stream error!";
      }
    }
  }
  exec_order_ += 1;
}
#endif
/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Checks dataset_sink_mode and generates the related error if any exist and calls
 * PreExecuteGraphDebugger.
 */
void DebugActor::DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                  const std::vector<AnfNodePtr> &origin_parameters_order,
                                  std::vector<DeviceContext *> device_contexts,
                                  OpContext<DeviceTensor> *const op_context, const AID *) {
  MS_LOG(INFO) << "Debug on step begin.";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  device_ctx_ = device_contexts[0];
  if (common::AnfAlgo::IsBackendGe()) {
    hooker::CheckDeprecatedDumpEnv();
    auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
    if ((profiler == nullptr || !profiler->IsInitialized()) &&
        device_ctx_->GetDeviceType() == device::DeviceType::kAscend) {
      auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      auto &hookDebugger = hooker::HookDebugger::GetInstance();
      hookDebugger.HookOnStepBegin(device_id, graphs, step_count_, false);
    }
    return;
  }
  if (DumpJsonParser::GetInstance().e2e_dump_enabled() && !graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    bool is_dataset_graph = graph->IsDatasetGraph();
    uint32_t cur_step = DumpJsonParser::GetInstance().cur_dump_iter();
    if (cur_step == 1 && DumpJsonParser::GetInstance().GetDatasetSink()) {
      uint32_t init_step = 0;
      DumpJsonParser::GetInstance().UpdateDumpIter(init_step);
      MS_LOG(INFO) << "In dataset sink mode, reset step to init_step: " << init_step;
    }
    DumpJsonParser::GetInstance().SetDatasetSink(is_dataset_graph);
  }
  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);
#ifdef ENABLE_DEBUGGER
  if (!graphs.empty()) {
    // First graph is the dataset graph when dataset_sink_mode = True
    auto graph = graphs[0];
    std::string error_info = CheckDatasetSinkMode(graph);
    if (!error_info.empty()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), error_info);
    }
  }
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr && debugger->DebuggerBackendEnabled()) {
    debugger->PreExecuteGraphDebugger(graphs, origin_parameters_order);
  }
#endif
  if (DumpJsonParser::GetInstance().e2e_dump_enabled()) {
    DumpJsonParser::GetInstance().ClearGraph();
    if (graphs.size() != device_contexts.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context), "Graph num:" + std::to_string(graphs.size()) +
                                                         " is not equal to device context size:" +
                                                         std::to_string(device_contexts.size()) + " for debug actor.");
    }
    for (size_t i = 0; i < graphs.size(); ++i) {
      MS_EXCEPTION_IF_NULL(graphs[i]);
      MS_EXCEPTION_IF_NULL(device_contexts[i]);
      if (device_contexts[i]->GetDeviceType() == device::DeviceType::kCPU) {
        DumpJsonParser::GetInstance().SaveGraph(graphs[i].get());
      }
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Dump parameters and constants and update dump iter for CPU. Call PostExecuteGraph Debugger for GPU and
 * Ascend and update step number of online debugger GPU.
 */
void DebugActor::DebugOnStepEnd(OpContext<DeviceTensor> *const, const AID *, int total_running_count_, int sink_size_) {
  MS_LOG(INFO) << "Debug on step end. total_running_count is: " << total_running_count_;
  auto context = MsContext::GetInstance();
  auto is_kbyk = context->IsKByKExecutorMode();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count_ = total_running_count_;
  if (dump_flag_ == true) {
    if (sink_size_ != 1 && !is_kbyk) {
      MS_EXCEPTION(ValueError) << "When using acl dump in data sink mode, sink size must be 1, but got " << sink_size_
                               << ".";
    }
    dump_flag_ = false;
  }

  auto &hookDebugger = hooker::HookDebugger::GetInstance();
  hookDebugger.HookOnStepEnd();
  device_ctx_->device_res_manager_->SyncAllStreams();
  std::lock_guard<std::mutex> locker(debug_mutex_);

  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    if (common::AnfAlgo::IsBackendGe()) {
      MS_LOG(INFO) << "Not kernel mode, skip post actions.";
      return;
    }
    // Reset exec_order for the next step
    exec_order_ = 0;
    debugger->Debugger::PostExecuteGraphDebugger();
  }
  DumpJsonParser::GetInstance().UpdateDumpIter(step_count_);
  MS_LOG(INFO) << "UpdateDumpIter: " << step_count_;
#endif
}

void DebugActor::Finalize() {
  DumpJsonParser::GetInstance().PrintUnusedKernel();
#ifdef ENABLE_DEBUGGER
  datadump::CheckOverflowKernel::ClearMemoryCache();
#endif
}
}  // namespace runtime
}  // namespace mindspore
