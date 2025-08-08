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

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "backend/backend_manager/backend_manager.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/kernel_graph/session_factory.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/backend/kernel_graph.h"
#include "include/common/runtime_conf/thread_bind_core.h"
#include "debug/profiler/profiler.h"

#include "backend/ms_infer_backend/ms_infer_backend.h"
#include "backend/ms_infer_backend/host_value_store.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

BackendGraphId MSInferBackend::backend_graph_id_ = 0;

namespace {
KernelGraphPtr OptimizeMindIR(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  auto session = session::SessionFactory::Get().Create(kSessionBasic);
  std::vector<KernelGraphPtr> kernel_graphs;
  auto kernel_graph =
    session->ConstructKernelGraph(func_graph, &kernel_graphs, device_context->GetDeviceType(), backend_jit_config);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Constructed kernel graph: " << kernel_graph->ToString()
               << " from func graph: " << func_graph->ToString();

  opt::OptimizationWithoutBackend(kernel_graph);
  auto kernel_executor = device_context->GetKernelExecutor();
  MS_EXCEPTION_IF_NULL(kernel_executor);
  kernel_executor->AddMindIRPass(kernel_graph);
  kernel_graph->SetInputNodes();
  return kernel_graph;
}
}  // namespace

BackendGraphId MSInferBackend::Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "MSInferBackend start build graph";

  auto kernel_graph = OptimizeMindIR(func_graph, backend_jit_config);
  auto graph_adapter = std::make_shared<GraphAdapter>(kernel_graph);
  MS_EXCEPTION_IF_NULL(graph_adapter);
  graph_adapter_map_[backend_graph_id_] = graph_adapter;

  graph_adapter->ConvertGraph();

  MS_LOG(INFO) << "MSInferBackend build graph success";

  return backend_graph_id_++;
}

RunningStatus MSInferBackend::Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kBackendGraphRunInner,
                                     std::to_string(graph_id), true);
  BindCoreForMainThread();

  auto graph_adapter_iter = graph_adapter_map_.find(graph_id);
  if (graph_adapter_iter == graph_adapter_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find graph id " << graph_id;
  }
  auto graph_adapter = graph_adapter_iter->second;

  // release python gil
  mindspore::ScopedLongRunning long_running;

  MS_LOG(INFO) << "MSInferBackend start run graph";

  graph_adapter->RunGraph(inputs, outputs);

  MS_LOG(INFO) << "MSInferBackend run graph end";

  return RunningStatus::kRunningSuccess;
}

void MSInferBackend::BindCoreForMainThread() {
  static bool is_bind_core_ = false;
  if (is_bind_core_) {
    return;
  }
  auto &bind_core_manager = runtime::ThreadBindCore::GetInstance();
  if (!bind_core_manager.is_enable_thread_bind_core_) {
    return;
  }

  const auto &core_list = bind_core_manager.get_thread_bind_core_list(runtime::kBindCoreModule::kMAIN);
  if (core_list.empty()) {
    MS_LOG(WARNING) << "Failed to bind thread core as no available core assigned to Main thread.";
  } else {
    bind_core_manager.bind_thread_core(core_list);
  }
  is_bind_core_ = true;
}

std::string MSInferBackend::ExportIR(const FuncGraphPtr &func_graph, const std::string &file_name, bool is_save_to_file,
                                     IRFormat ir_format) {
  return "";
}

void MSInferBackend::ConvertIR(const FuncGraphPtr &func_graph,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                               IRFormat ir_format) {}

void MSInferBackend::Clear() { HostValueStore::GetInstance().Clear(); }

MS_REGISTER_BACKEND(kMSInferBackendName, MSInferBackend)

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
