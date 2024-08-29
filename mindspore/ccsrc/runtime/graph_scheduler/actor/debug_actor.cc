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
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "utils/ms_context.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#include "debug/debugger/debugger_utils.h"
#include "debug/hooker/hook_debugger.h"
#endif
#include "debug/data_dump/data_dumper.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "op_def/nn_op_name.h"
#include "debug/data_dump/overflow_counter.h"

namespace mindspore {
namespace runtime {
void DebugActor::ACLDump(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs, bool is_kbyk) {
  std::vector<std::string> all_kernel_names;
  std::vector<std::string> set_dump_names;
  for (const auto &graph : graphs) {
    auto all_kernels = graph->execution_order();
    std::for_each(all_kernels.begin(), all_kernels.end(), [&](const auto &k) {
      all_kernel_names.push_back(k->fullname_with_scope());
      auto dump_flag = common::AnfAlgo::GetDumpFlag(k);
      if (dump_flag.has_value() && dump_flag.value().compare("true") == 0) {
        set_dump_names.push_back(k->fullname_with_scope());
      }
    });
  }

  auto step_count_num = 0;
  step_count_num = step_count_;
  if (step_count_ == 1 && is_dataset_sink_ == 1) {
    step_count_num = 0;
  }
  if (!graphs.empty()) {
    auto graph = graphs[0];
    is_dataset_sink_ = graph->IsDatasetGraph();
  }
  auto enable_ge_dump = common::GetEnv("ENABLE_MS_GE_DUMP");
  if (DumpJsonParser::GetInstance().async_dump_enabled() &&
      ((DumpJsonParser::GetInstance().DumpEnabledForIter() && is_kbyk) || (enable_ge_dump != "1" && !is_kbyk))) {
    bool is_init = false;
    if ((enable_ge_dump != "1") && !(DumpJsonParser::GetInstance().DumpEnabledForIter())) {
      is_init = true;
    } else {
      std::string dump_path = DumpJsonParser::GetInstance().path();
      std::string dump_path_step = dump_path + "/" + std::to_string(step_count_num);
      auto real_path = FileUtils::CreateNotExistDirs(dump_path_step, false);
      if (!real_path.has_value()) {
        MS_LOG(WARNING) << "Fail to create acl dump dir " << real_path.value();
        return;
      }
    }
    dump_flag_ = true;
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      registered_dumper->Initialize();
      if (DumpJsonParser::GetInstance().dump_mode() ==
          static_cast<uint32_t>(mindspore::DumpJsonParser::JsonDumpMode::DUMP_KERNELS_WITH_FLAG)) {
        if (set_dump_names.empty()) {
          MS_LOG(WARNING) << "[set dump] There is no target with dump flag.";
          set_dump_names.push_back("NoSetDumpTarget");
        }
        registered_dumper->EnableDump(device_id, step_count_num, is_init, set_dump_names);
      } else {
        registered_dumper->EnableDump(device_id, step_count_num, is_init, all_kernel_names);
      }
    }
  }
}

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
#ifndef ENABLE_SECURITY
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
#endif
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
  auto is_kbyk = context->IsKByKExecutorMode();
  auto backend = context->backend_policy();
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if ((profiler == nullptr || !profiler->IsInitialized()) &&
      device_ctx_->GetDeviceType() == device::DeviceType::kAscend) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    if (common::GetEnv("ENABLE_MS_GE_DUMP") != "1") {
      ACLDump(device_id, graphs, is_kbyk);
    }
    HandleHookDebugger(device_id, graphs, is_kbyk);
  }

  if (IsE2EDumpEnabled() && !graphs.empty()) {
    HandleE2EDump(graphs);
  }

  if (backend == "ge") {
    return;
  }

  MS_EXCEPTION_IF_NULL(op_context);
  std::lock_guard<std::mutex> locker(debug_mutex_);

  HandleDebugger(graphs, origin_parameters_order, op_context);

  if (IsE2EDumpEnabled()) {
    ClearAndSaveGraphs(graphs, device_contexts, op_context);
  }
}

void DebugActor::HandleHookDebugger(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs, bool is_kbyk) {
#ifdef ENABLE_DEBUGGER
  auto &hookDebugger = hooker::HookDebugger::GetInstance();
  hookDebugger.HookOnStepBegin(device_id, graphs, step_count_, is_dataset_sink_, is_kbyk);
#endif
}

bool DebugActor::IsE2EDumpEnabled() {
#ifndef ENABLE_SECURITY
  return DumpJsonParser::GetInstance().e2e_dump_enabled();
#else
  return false;
#endif
}

void DebugActor::HandleE2EDump(const std::vector<KernelGraphPtr> &graphs) {
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

void DebugActor::HandleDebugger(const std::vector<KernelGraphPtr> &graphs,
                                const std::vector<AnfNodePtr> &origin_parameters_order,
                                OpContext<DeviceTensor> *const op_context) {
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
}

void DebugActor::ClearAndSaveGraphs(const std::vector<KernelGraphPtr> &graphs,
                                    const std::vector<DeviceContext *> &device_contexts,
                                    OpContext<DeviceTensor> *const op_context) {
#ifndef ENABLE_SECURITY
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
#endif
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Dump parameters and constants and update dump iter for CPU. Call PostExecuteGraph Debugger for GPU and
 * Ascend and update step number of online debugger GPU.
 */
void DebugActor::DebugOnStepEnd(OpContext<DeviceTensor> *const, const AID *, int total_running_count_) {
  MS_LOG(INFO) << "Debug on step end. total_running_count is: " << total_running_count_;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string backend = context->backend_policy();
  step_count_ = total_running_count_;
  device_ctx_->device_res_manager_->SyncAllStreams();
  if (dump_flag_ == true) {
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      registered_dumper->Finalize();
    }
    dump_flag_ = false;
  }

#ifdef ENABLE_DEBUGGER
  auto &hookDebugger = hooker::HookDebugger::GetInstance();
  hookDebugger.HookOnStepEnd();
#endif

  std::lock_guard<std::mutex> locker(debug_mutex_);

#ifndef ENABLE_SECURITY
  if (DumpJsonParser::GetInstance().GetIterDumpFlag()) {
    CPUE2eDump::DumpParametersData();
    CPUE2eDump::DumpConstantsData();
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  if (debugger != nullptr) {
    if (backend == "ge" && !debugger->GetAscendKernelByKernelFlag()) {
      MS_LOG(INFO) << "Not kernel mode, skip post actions.";
      return;
    }
    // Reset exec_order for the next step
    exec_order_ = 0;
    debugger->Debugger::PostExecuteGraphDebugger();
  }
#ifndef ENABLE_SECURITY
  DumpJsonParser::GetInstance().UpdateDumpIter(step_count_);
  MS_LOG(INFO) << "UpdateDumpIter: " << step_count_;
#endif
#endif
}
}  // namespace runtime
}  // namespace mindspore
