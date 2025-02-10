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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GEBACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GEBACKEND_H_

#include <memory>
#include "backend/backend_manager/backend_manager.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
// The base class of all supported backend.
class BACKEND_EXPORT GEBackend : public BackendBase {
 public:
  GEBackend() = default;
  ~GEBackend() = default;

  // The backend graph Build interface, the return value is the built graph id.
  BackendGraphId Build(const FuncGraphPtr &func_graph) override { return 0; }

  // The backend graph Run interface by the graph_id which are generated through the graph Build interface above.
  RunningStatus Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) { return kRunningSuccess; }
};

using GEBackendPtr = std::shared_ptr<GEBackend>;
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_GEBACKEND_H_
