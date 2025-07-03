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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_INFER_BACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_INFER_BACKEND_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "include/backend/visible.h"
#include "backend/backend_manager/backend_base.h"
#include "backend/backend_manager/backend_jit_config.h"
#include "ir/tensor.h"

#include "backend/ms_infer_backend/graph_adapter.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

class BACKEND_EXPORT MSInferBackend : public BackendBase {
 public:
  MSInferBackend() = default;
  ~MSInferBackend() = default;

  // The backend graph Build interface, the return value is the built graph id.
  BackendGraphId Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) override;

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) override;

  std::string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                       IRFormat ir_format) override;

  void ConvertIR(const FuncGraphPtr &anf_graph,
                 const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                 IRFormat ir_format) override;

  void Clear() override;

 private:
  static BackendGraphId backend_graph_id_;
  BackendJitConfig backend_jit_config_;
  std::unordered_map<BackendGraphId, GraphAdapterPtr> graph_adapter_map_;
};

using MSInferBackendPtr = std::shared_ptr<MSInferBackend>;

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_MS_INFER_BACKEND_H_
