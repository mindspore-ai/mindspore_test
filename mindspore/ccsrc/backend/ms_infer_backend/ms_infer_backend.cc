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
#include <utility>

#include "backend/backend_manager/backend_manager.h"

#include "backend/ms_infer_backend/ms_infer_backend.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

BackendGraphId MSInferBackend::backend_graph_id_ = 0;

BackendGraphId MSInferBackend::Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);

  auto compiled_graph = CompileGraph(func_graph, backend_jit_config);

  auto graph_adapter = std::make_shared<GraphAdapter>(compiled_graph);
  MS_EXCEPTION_IF_NULL(graph_adapter);
  graph_adapter_map_[backend_graph_id_] = graph_adapter;

  graph_adapter->ConvertGraph();

  return backend_graph_id_++;
}

KernelGraphPtr MSInferBackend::CompileGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  bool is_pynative = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;

  auto session = session::SessionFactory::Get().Create(kSessionBasic);
  auto device_target = device_context->GetDeviceType();
  std::vector<KernelGraphPtr> kernel_graphs;
  auto kernel_graph = session->ConstructKernelGraph(func_graph, &kernel_graphs, device_target, backend_jit_config);

  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Compile graph: " << kernel_graph->ToString() << ", kernel graph";

  kernel_graph->SetExecOrderByDefault();
  kernel_graph->set_flag(kFlagPyNativeRunInGraph, is_pynative);

  auto io_nodes = std::make_pair(kernel_graph->inputs(), kernel_graph->outputs());
  (void)graph_compiler_->CompileGraph(kernel_graph, io_nodes, device_context, device::RunMode::kKernelMode,
                                      is_pynative);
  return kernel_graph;
}

RunningStatus MSInferBackend::Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  auto graph_adapter_iter = graph_adapter_map_.find(graph_id);
  if (graph_adapter_iter == graph_adapter_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find graph id " << graph_id;
  }
  auto graph_adapter = graph_adapter_iter->second;

  graph_adapter->RunGraph(inputs, outputs);

  return RunningStatus::kRunningSuccess;
}

std::string MSInferBackend::ExportIR(const FuncGraphPtr &func_graph, const std::string &file_name, bool is_save_to_file,
                                     IRFormat ir_format) {
  return "";
}

void MSInferBackend::ConvertIR(const FuncGraphPtr &func_graph,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                               IRFormat ir_format) {}

void MSInferBackend::Clear() {}

MS_REGISTER_BACKEND(kMSInferBackendName, MSInferBackend)

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
