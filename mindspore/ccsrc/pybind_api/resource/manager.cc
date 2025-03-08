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

#include "pybind_api/resource/manager.h"

#include <memory>
#include <map>
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/pipeline_jit.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pynative/pynative_execute.h"
#include "pynative/op_function/converter.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/expander/utils.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils.h"
#include "include/backend/debug/execute_order_tracker/execute_order_tracker.h"
#include "utils/log_adapter.h"
#include "utils/info.h"
#include "include/common/utils/comm_manager.h"
#include "utils/interpret_node_recorder.h"
#include "include/common/debug/dump_proto.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "backend/common/session/executor_manager.h"
#include "backend/common/session/session_factory.h"
#include "backend/backend_manager/backend_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/stream_synchronizer.h"
#include "debug/profiler/profiler.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "debug/profiler/profiling.h"
#include "include/backend/debug/tft_adapter/tft_wait_sem.h"
#include "kernel/graph_kernel/graph_kernel_builder_manager.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "include/common/symbol_engine/symbol_engine_impl.h"
#include "pipeline/jit/ps/pass_config.h"

#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "abstract/abstract_value.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/embedding_cache/data_queue_manager.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "ir/cell.h"
#endif

#include "frontend/ir/py_execute_py.h"  // Only include one-time in the whole project.
#include "mindspore/ccsrc/pynative/op_function/auto_generate/tensor_func_utils.h"
#include "backend/common/somas/somas.h"

namespace mindspore {
void RecordExitStatus() { MS_LOG(INFO) << "Status record: system exit."; }

void MemoryRecycle() {
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif
  pipeline::ReclaimOptimizer();
  session::ExecutorManager::Instance().ClearDoneTasks();
  ad::g_k_prims.clear();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
  pipeline::CleanCache();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  ConfigManager::GetInstance().ResetConfig();
  ScopeManager::GetInstance().ClearScope();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  FuncGraphLoopBreaker::Inst().BreakLoop();
}

void ClearResPart1() {
  pynative::PyNativeExecutor::GetInstance()->WorkerJoin();
  runtime::OpExecutor::GetInstance().WorkerJoin();
  // When the python process exits, the kernels on the device may not have finished executing.
  device::KernelRuntimeManager::Instance().WaitTaskFinishOnDevice();
  device::DeviceContextManager::GetInstance().WaitTaskFinishOnDevice();
  tensor::StubTensorConverter::GetInstance().Clear();
  RecordExitStatus();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
  mindspore::RDR::ResetRecorder();
#endif
  runtime::GraphScheduler::GetInstance().Clear();
  runtime::ProfilerAnalyzer::GetInstance().Clear();
  opt::PassConfigure::Instance().Clear();

  MS_LOG(INFO) << "Start Finalize StreamSynchronizer...";
  device::StreamSynchronizer::GetInstance()->Finalize();
  MS_LOG(INFO) << "End Finalize StreamSynchronizer...";

  PrimitivePy::ClearHookRes();
  ad::g_k_prims.clear();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();

  abstract::ClearPrimEvaluatorMap();
  pipeline::GetMethodMap().clear();
  pipeline::GetAttrMap().clear();
#ifdef WITH_BACKEND
  pipeline::GraphExecutorPy::GetInstance()->ClearInfo();
  pipeline::JitExecutorPy::GetInstance()->ClearInfo();
#endif
  pipeline::GraphExecutorPy::ClearRes();
  pipeline::JitExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
}

void ClearResPart2() {
  MS_LOG(INFO) << "Start clear PyNativeExecutor...";
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  MS_LOG(INFO) << "End clear PyNativeExecutor.";

  MS_LOG(INFO) << "Start clear ConfigManager...";
  ConfigManager::GetInstance().ResetIterNum();
  MS_LOG(INFO) << "End clear ConfigManager.";

  session::ExecutorManager::Instance().Clear();

  MS_LOG(INFO) << "Start clear device context...";
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  MS_LOG(INFO) << "End clear device context.";

  MS_LOG(INFO) << "Start clear kernel runtime...";
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  MS_LOG(INFO) << "End clear kernel runtime.";

  MS_LOG(INFO) << "Start clear CollectiveManager...";
  // for GE, HcclCommDestroy should after RemoveGraph in ClearGraphWrapper in ClearDeviceContexts
  (void)distributed::collective::CollectiveManager::instance()->Finalize();
  MS_LOG(INFO) << "End clear CollectiveManager.";

  MS_LOG(INFO) << "Start clear AnalysisResultCacheMgr...";
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  MS_LOG(INFO) << "End clear AnalysisResultCacheMgr.";

  MS_LOG(INFO) << "Start clear AnalysisContext...";
  abstract::AnalysisContext::ClearContext();
  MS_LOG(INFO) << "End clear AnalysisContext...";

  MS_LOG(INFO) << "Start clear AnalysisSchedule...";
  abstract::AnalysisSchedule::GetInstance().Stop();
  MS_LOG(INFO) << "End clear AnalysisSchedule...";
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->Reset();
#endif
  pipeline::CleanCache();
}

void ClearResPart3() {
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  MS_LOG(INFO) << "Start clear ClearObjectCache...";
  parse::data_converter::ClearObjectCache();
  MS_LOG(INFO) << "End clear ClearObjectCache...";

  MS_LOG(INFO) << "Start clear Parser...";
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "End clear Parser...";

  MS_LOG(INFO) << "Start ClearTraceStack...";
  trace::ClearTraceStack();
  MS_LOG(INFO) << "End ClearTraceStack...";

  MS_LOG(INFO) << "Start clear InterpretNodeRecorder...";
  InterpretNodeRecorder::GetInstance().Clear();
  MS_LOG(INFO) << "End clear InterpretNodeRecorder...";

  MS_LOG(INFO) << "Start clear parallel::entire_costgraph...";
  parallel::entire_costgraph.reset();
  MS_LOG(INFO) << "End clear parallel::entire_costgraph...";

  MS_LOG(INFO) << "Start clear ProtobufLibrary...";
  google::protobuf::ShutdownProtobufLibrary();
  MS_LOG(INFO) << "End clear ProtobufLibrary...";

  MS_LOG(INFO) << "Start clear ParserDefaultObjects ...";
  pynative::ParserDefaultObjects::GetInstance().ClearRes();
  MS_LOG(INFO) << "End clear ParserDefaultObjects...";

  // ResetPythonScope after all py::object is freed.
  MS_LOG(INFO) << "Start clear python_adapter...";
  python_adapter::ResetPythonScope();
  MS_LOG(INFO) << "End clear python_adapter.";
}

void ClearSingleton() {
  MS_LOG(INFO) << "Start clear singleton...";
  profiler::Profiler::Clear();
  debug::tft::TFTWaitSem::GetInstance().Clear();
#ifdef ENABLE_AKG
  kernel::GraphKernelBuildManager::Instance().Clear();
#endif
  somas::SomasManager::Instance().Clear();
  GraphKernelInfoManager::Instance().Clear();
  device::DataQueueMgr::GetInstance().Clear();
  session::SessionFactory::Get().Clear();
  device::KernelRuntimeManager::Instance().Clear();
  backend::BackendManager::GetInstance().Clear();
  ExecuteOrderTracker::GetInstance().Clear();
  OpPrimPyRegister::GetInstance().Clear();
  DumpJsonParser::Finalize();
  CommManager::Clear();
  expander::ClearAllCache();

  MS_LOG(INFO) << "End clear singleton.";
}

void ClearResAtexit() {
  MS_LOG(INFO) << "Pipeline clear all resource";
  try {
    MsException::Instance().CheckException();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Check exception before process exit: " << e.what();
  }
  ClearResPart1();
  ClearResPart2();

  mindspore::trans::FormatHelper::GetInstance().Clear();
  ClearResPart3();
  ClearSingleton();
  //  The premature unloading of the plugin .so triggers the process to exit during the termination phase. Other
  //  components' singletons, static variables, and global variables in MindSpore may inadvertently invoke the plugin
  //  interface, resulting in an undefined coredump.
}
}  // namespace mindspore
