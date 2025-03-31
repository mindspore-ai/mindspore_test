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
#ifndef _WIN32
#include <libgen.h>
#endif
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"

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

std::string GetBackendLibNameByType(BackendType backend_type) {
  auto iter = backend_type_to_lib_name.find(backend_type);
  if (iter == backend_type_to_lib_name.end()) {
    MS_LOG(EXCEPTION) << "Invalid backend type for the dynamic load: " << backend_type;
  }
  return iter->second;
}

BackendType GetBackendType(const std::string &backend_name) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return kMSBackend;
  }

  if (!backend_name.empty()) {
    return GetBackendTypeByName(backend_name);
  }

  if (context->IsKByKExecutorMode()) {
    return kMSBackend;
  } else {
    return kGEBackend;
  }
}

std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string curr_so_path = dl_info.dli_fname;
  return dirname(curr_so_path.data());
#else
  return "";
#endif
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
    if (backends_[i] != nullptr) {
      backends_[i]->Clear();
      backends_[i] = nullptr;
    }
  }

  backend_creators_.clear();
  backend_load_handle_.clear();
}

std::pair<BackendType, BackendGraphId> BackendManager::Build(const FuncGraphPtr &func_graph,
                                                             const BackendJitConfig &backend_jit_config,
                                                             const std::string &backend_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  auto graph_id = backend->Build(func_graph, backend_jit_config);
  MS_LOG(INFO) << "Backend build graph, backend name: " << backend_name << ", backend type: " << backend_type
               << ", backend graph id: " << graph_id;
  return {backend_type, graph_id};
}

RunningStatus BackendManager::Run(BackendType backend_type, BackendGraphId graph_id, const VectorRef &inputs,
                                  VectorRef *outputs) {
  auto backend = backends_[backend_type];
  MS_EXCEPTION_IF_NULL(backend);
  MS_LOG(INFO) << "Backend run graph: " << graph_id << ", backend type: " << backend_type;
  return backend->Run(graph_id, inputs, outputs);
}

string BackendManager::ExportIR(const FuncGraphPtr &anf_graph, const std::string &file_name, bool is_save_to_file,
                                IRFormat ir_format, const std::string &backend_name) {
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ExportIR(anf_graph, file_name, is_save_to_file, ir_format);
}

void BackendManager::ConvertIR(const FuncGraphPtr &anf_graph,
                               const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors,
                               IRFormat ir_format, const std::string &backend_name) {
  auto backend_type = GetBackendType(backend_name);
  auto backend = GetOrCreateBackend(backend_type);
  MS_EXCEPTION_IF_NULL(backend);
  return backend->ConvertIR(anf_graph, init_tensors, ir_format);
}

void BackendManager::LoadBackend(BackendType backend_type) {
  if (backend_load_handle_.count(backend_type) > 0) {
    return;
  }
  if (backend_type != kGEBackend) {
    MS_LOG(EXCEPTION) << "Only the ge backend support the dynamic load. ";
  }

  std::string backend_lib_name = GetBackendLibNameByType(backend_type);
  MS_LOG(INFO) << "Backendmanager dlopen backend lib name: " << backend_lib_name;
  void *handle;
  std::string err_msg = "";
#ifndef _WIN32
  std::string cur_backend_lib_name = GetCurrentDir() + "/" + backend_lib_name;
  MS_LOG(INFO) << "Backendmanager dlopen current backend lib name: " << cur_backend_lib_name;
  handle = dlopen(cur_backend_lib_name.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#else
  handle = LoadLibrary(backend_lib_name.c_str());
  err_msg = std::to_string(GetLastError());
#endif

  if (handle == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + backend_lib_name + " failed. Error: " + err_msg;
  }
  (void)backend_load_handle_.emplace(backend_type, handle);
}

void BackendManager::UnloadBackend() {
  for (auto iter : backend_load_handle_) {
    auto backend_lib_name = GetBackendLibNameByType(iter.first);
    auto handle = iter.second;
#ifndef _WIN32
    if (dlclose(handle) != 0) {
      MS_LOG(EXCEPTION) << "Closing " + backend_lib_name + " handle failed. Error: " + GetDlErrorMsg();
    }
#else
    if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(handle))) {
      MS_LOG(EXCEPTION) << "Closing " + backend_lib_name + " handle failed. Error: " + std::to_string(GetLastError());
    }
#endif
  }
}

BackendBase *BackendManager::GetOrCreateBackend(BackendType backend_type) {
  if (backends_[backend_type] != nullptr) {
    return backends_[backend_type].get();
  }

  // Only the ge backend support the dynamic load.
  if (backend_type == kGEBackend) {
    LoadBackend(backend_type);
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
