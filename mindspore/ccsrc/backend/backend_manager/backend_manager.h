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
#ifndef MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDMANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDMANAGER_H_

#include <memory>
#include <string>
#include <utility>
#include <map>
#include <vector>
#include "backend/backend_manager/backend_base.h"

namespace mindspore {
namespace backend {
// The register entry of new backend.
#define MS_REGISTER_BACKEND(BACKEND_NAME, BACKEND_CLASS)                    \
  static const BackendRegister g_backend_##BACKEND_NAME##_reg(BACKEND_NAME, \
                                                              []() { return std::make_shared<BACKEND_CLASS>(); });
using BackendCreator = std::function<std::shared_ptr<BackendBase>()>;

// The backend name must be equal to the backend field of api "mindspore.jit".
const char kMSBackendName[] = "ms_backend";
const char kGEBackendName[] = "GE";

// The name of backend lib.
const char kGEBackendLibName[] = "libmindspore_ge_backend.so";

// The backend type enum, please add a new enumeration definition before kInvalidBackend when adding a new backend.
enum BackendType {
  kMSBackend = 0,
  kGEBackend,
  kInvalidBackend,
};

const std::map<std::string, BackendType> backend_name_to_type = {{kMSBackendName, kMSBackend},
                                                                 {kGEBackendName, kGEBackend}};
const std::map<BackendType, std::string> backend_type_to_name = {{kMSBackend, kMSBackendName},
                                                                 {kGEBackend, kGEBackendName}};

const std::map<BackendType, std::string> backend_type_to_lib_name = {{kGEBackend, kGEBackendLibName}};

class BACKEND_MANAGER_EXPORT BackendManager {
 public:
  static BackendManager &GetInstance();
  // Record the BackendCreator by the backend name.
  void Register(const std::string &backend_name, BackendCreator &&backend_creator);

  std::vector<GraphFragmentPtr> Split(const FuncGraphPtr &func_graph, const std::string &backend_name);

  // The processing entry of graph building by the given backend.
  // The return value are the selected backend type and the built graph id.
  std::pair<BackendType, BackendGraphId> Build(const FuncGraphPtr &func_graph,
                                               const BackendJitConfig &backend_jit_config,
                                               const std::string &backend_name = "");

  // The processing entry of graph running by the backend_type and graph_id
  // which are generated through the graph Build interface above.
  RunningStatus Run(BackendType backend_type, BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs);

  // Clear the members.
  void Clear();

  // convert mindir to ir_format
  void ConvertIR(const FuncGraphPtr &anf_graph,
                 const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors, IRFormat ir_format,
                 const std::string &backend_name = "");

  // export graph to ir_format. If is_save_to_file=True, save as file; if False, return as string
  string ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file, IRFormat ir_format,
                  const std::string &backend_name = "");

  void SetPyBoostRegistered(const IsPyBoostRegisteredFunc &func, const RunPyBoostCallFunc &call_func) {
    func_ = func;
    call_func_ = call_func;
  }

 private:
  BackendManager() = default;
  ~BackendManager() = default;

  void LoadBackend(BackendType backend_type);
  void UnloadBackend();

  BackendBase *GetOrCreateBackend(BackendType backend_type);

  IsPyBoostRegisteredFunc func_;
  RunPyBoostCallFunc call_func_;

  // BackendType -> BackendLoadHandle.
  std::map<BackendType, void *> backend_load_handle_;

  // BackendType -> BackendCreator.
  std::map<BackendType, BackendCreator> backend_creators_;

  BackendBasePtr backends_[kInvalidBackend];
};

class BackendRegister {
 public:
  BackendRegister(const std::string &backend_name, BackendCreator &&backend_creator) {
    BackendManager::GetInstance().Register(backend_name, std::move(backend_creator));
  }
  ~BackendRegister() = default;
};
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_BACKENDMANAGER_BACKENDMANAGER_H_
