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

#include "backend/ms_infer_backend/ms_infer_backend.h"
#include "backend/ms_infer_backend/host_value_store.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

BackendGraphId MSInferBackend::backend_graph_id_ = 0;

BackendGraphId MSInferBackend::Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "MSInferBackend start build graph";

  auto graph_adapter = std::make_shared<GraphAdapter>(func_graph);
  MS_EXCEPTION_IF_NULL(graph_adapter);
  graph_adapter_map_[backend_graph_id_] = graph_adapter;

  // clear host value store before build new graph
  HostValueStore::GetInstance().Clear();

  graph_adapter->ConvertGraph();

  MS_LOG(INFO) << "MSInferBackend build graph success";

  return backend_graph_id_++;
}

RunningStatus MSInferBackend::Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
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
