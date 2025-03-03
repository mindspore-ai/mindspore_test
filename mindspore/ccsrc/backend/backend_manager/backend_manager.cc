/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "backend/backend_manager/backend_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace backend {
namespace {
BackendType GetBackendTypeByName(const std::string &backend_name) {
  auto iter = backend_name_to_type.find(backend_name);
  if (iter == backend_name_to_type.end()) {
    MS_LOG(EXCEPTION) << "Illegal backend name: " << backend_name;
  }
  return iter->second;
}

std::string GetBackendNameByType(BackendType backend_type) {
  auto iter = backend_type_to_name.find(backend_type);
  if (iter == backend_type_to_name.end()) {
    MS_LOG(EXCEPTION) << "Illegal backend type: " << backend_type;
  }
  return iter->second;
}

BackendType GetBackendType() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return kMSBackend;
  }

  if (context->IsKByKExecutorMode()) {
    return kMSBackend;
  } else {
    return kGEBackend;
  }
}
}  // namespace

BackendManager &BackendManager::GetInstance() {
  static BackendManager instance{};
  return instance;
}

void BackendManager::Register(const std::string &backend_name, BackendCreator &&backend_creator) {
  auto backend_type = GetBackendTypeByName(backend_name);
  if (backend_creators_.find(backend_type) == backend_creators_.end()) {
    (void)backend_creators_.emplace(backend_type, std::move(backend_creator));
  }
}

void BackendManager::Clear() {
  for (size_t i = 0; i < kInvalidBackend; i++) {
    backends_[i] = nullptr;
  }
}

std::pair<BackendType, BackendGraphId> BackendManager::Build(const FuncGraphPtr &func_graph,
                                                             const BackendJitConfig &backend_jit_config,
                                                             const std::string &backend_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  BackendType backend_type = kInvalidBackend;
  if (backend_name.empty()) {
    backend_type = GetBackendType();
  } else {
    backend_type = GetBackendTypeByName(backend_name);
  }

  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  auto graph_id = backend->Build(func_graph, backend_jit_config);
  return {backend_type, graph_id};
}

RunningStatus BackendManager::Run(BackendType backend_type, BackendGraphId graph_id, const VectorRef &inputs,
                                  VectorRef *outputs) {
  auto backend = backends_[backend_type];
  MS_EXCEPTION_IF_NULL(backend);
  return backend->Run(graph_id, inputs, outputs);
}

string BackendManager::ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                                IRFormat ir_format, const std::string &backend_name) {
  BackendType backend_type = kInvalidBackend;
  if (backend_name.empty()) {
    backend_type = GetBackendType();
  } else {
    backend_type = GetBackendTypeByName(backend_name);
  }

  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ExportIR(anf_graph, file_name, is_save_to_file, ir_format);
}

void BackendManager::ConvertIR(const FuncGraphPtr &anf_graph,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                               IRFormat ir_format, const std::string &backend_name) {
  BackendType backend_type = kInvalidBackend;
  if (backend_name.empty()) {
    backend_type = GetBackendType();
  } else {
    backend_type = GetBackendTypeByName(backend_name);
  }

  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ConvertIR(anf_graph, init_tensors, ir_format);
}

BackendBase *BackendManager::GetOrCreateBackend(BackendType backend_type) {
  if (backends_[backend_type] != nullptr) {
    return backends_[backend_type].get();
  }

  auto creator_iter = backend_creators_.find(backend_type);
  if (creator_iter == backend_creators_.end()) {
    MS_LOG(EXCEPTION) << "Create backend failed, please make sure the backend:" << GetBackendNameByType(backend_type)
                      << " has been registered.";
  }

  MS_LOG(INFO) << "The created backend type: " << backend_type;
  auto backend = (creator_iter->second)();
  MS_EXCEPTION_IF_NULL(backend);
  backend->SetPyBoostRegistered(func_, call_func_);
  backends_[backend_type] = backend;
  return backend.get();
}
}  // namespace backend
}  // namespace mindspore
