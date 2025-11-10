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

#include "kernel/cpu/custom/kernel_mod_impl/op_plugin_utils.h"
#include <unordered_set>
#if defined(_WIN32)
#include <windows.h>
#define DL_OPEN(path)                                                                                   \
  [](const std::string &p) -> void * {                                                                  \
    SetLastError(0);                                                                                    \
    return reinterpret_cast<void *>(LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)); \
  }(path)

#define DL_SYM(handle, name)                                                     \
  [](void *h, const char *n) -> void * {                                         \
    SetLastError(0);                                                             \
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(h), n)); \
  }(handle, name)

#define DL_CLOSE(handle) FreeLibrary((HMODULE)handle)

#define DL_ERROR()                                  \
  []() -> const char * {                            \
    static std::string errMsg;                      \
    DWORD errCode = GetLastError();                 \
    if (errCode == 0) return nullptr;               \
    errMsg = "WinError " + std::to_string(errCode); \
    return errMsg.c_str();                          \
  }()
#elif !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#define DL_OPEN(path) dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)
#define DL_SYM(handle, name) dlsym(handle, name)
#define DL_CLOSE(handle) dlclose(handle)
#define DL_ERROR() dlerror()
#endif
#include <string>
#include <algorithm>
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::kernel {
namespace op_plugin {
void *GetOpPluginHandle() {
  static bool is_initialized = false;
  static void *handle = nullptr;
  if (is_initialized) {
    return handle;
  }

  is_initialized = true;
  const char *op_plugin_path = common::EnvHelper::GetInstance()->GetEnv("MS_OP_PLUGIN_PATH");

  if (op_plugin_path == nullptr) {
    MS_LOG(INFO) << "MS_OP_PLUGIN_PATH is not set. Op plugin will not be loaded.";
    return nullptr;
  }

  auto real_path = FileUtils::GetRealPath(op_plugin_path).value_or("");
  if (real_path.empty()) {
    MS_LOG(ERROR) << "Failed to resolve the real path for MS_OP_PLUGIN_PATH: " << op_plugin_path;
    return nullptr;
  }
  handle = DL_OPEN(real_path);
  if (handle == nullptr) {
    MS_LOG(WARNING) << "Failed to open op plugin file: " << real_path << " Error code: " << DL_ERROR();
  }

  return handle;
}

bool IsOpPluginKernel(const std::string &op_name) {
  static bool initialized = false;
  static bool (*reg_func)(const char *) = nullptr;
  if (!initialized) {
    initialized = true;
    void *handle = GetOpPluginHandle();
    if (handle == nullptr) {
      return false;
    }
    constexpr auto reg_func_name = "IsKernelRegistered";
    reg_func = reinterpret_cast<std::add_pointer<bool(const char *)>::type>(DL_SYM(handle, reg_func_name));
    if (reg_func == nullptr) {
      MS_LOG(WARNING) << "Error occurs when fetching function '" << reg_func_name
                      << "' from op plugin library. Error code: " << DL_ERROR();
      return false;
    }
  }
  return reg_func != nullptr && reg_func(op_name.c_str());
}
const std::unordered_set<std::string> &GetAllOpPluginKernelNames() {
  static bool initialized = false;
  static int (*get_op_count)() = nullptr;
  static char **(*get_all_ops)() = nullptr;
  static std::unordered_set<std::string> op_names;
  if (!initialized) {
    initialized = true;
    void *handle = GetOpPluginHandle();
    if (handle == nullptr) {
      return op_names;
    }
    constexpr auto get_op_count_func_name = "GetRegisteredOpCount";
    constexpr auto get_all_ops_func_name = "GetAllRegisteredOps";
    get_op_count = reinterpret_cast<std::add_pointer<int()>::type>(DL_SYM(handle, get_op_count_func_name));
    get_all_ops = reinterpret_cast<std::add_pointer<char **()>::type>(DL_SYM(handle, get_all_ops_func_name));
    if (get_op_count == nullptr || get_all_ops == nullptr) {
      MS_LOG(WARNING) << "Error occurs when fetching function '" << get_op_count_func_name << "' or '"
                      << get_all_ops_func_name << "' from op plugin library. Error code: " << DL_ERROR();
      return op_names;
    }
    int op_count = get_op_count();
    char **op_names_ptr = get_all_ops();
    for (int i = 0; i < op_count; ++i) {
      op_names.insert(std::string(op_names_ptr[i]));
    }
  }
  return op_names;
}

OpPluginKernelParam CreateOpPluginParam(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs,
                                        const std::vector<KernelTensor *> &workspace) {
  OpPluginKernelParam param;

  // Process inputs
  for (const auto &input : inputs) {
    param.params.push_back(input->device_ptr());
    const auto &in_shape = input->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(in_shape.data()));
    param.ndims.push_back(SizeToInt(in_shape.size()));
    param.dtype_strings.push_back(TypeIdToString(input->dtype_id(), true));
  }

  // Process outputs
  for (const auto &output : outputs) {
    param.params.push_back(output->device_ptr());
    const auto &out_shape = output->GetShapeVector();
    param.shapes.push_back(const_cast<int64_t *>(out_shape.data()));
    param.ndims.push_back(SizeToInt(out_shape.size()));
    param.dtype_strings.push_back(TypeIdToString(output->dtype_id(), true));
  }

  // Process workspace
  std::transform(workspace.begin(), workspace.end(), std::back_inserter(param.params),
                 [](const KernelTensor *ws) { return ws->device_ptr(); });

  param.dtypes.reserve(param.dtype_strings.size());
  std::transform(param.dtype_strings.begin(), param.dtype_strings.end(), std::back_inserter(param.dtypes),
                 [](const std::string &dtype_str) { return dtype_str.c_str(); });

  param.kernel_info.SetKernelInput(inputs);
  return param;
}

int LaunchOpPluginKernel(const std::string &op_name, size_t nparam, void **params, int *ndims, int64_t **shapes,
                         const char **dtypes, void *kernel_info, void *stream) {
  int (*op_plugin_func)(int, void **, int *, int64_t **, const char **, void *, void *) = nullptr;
  void *handle = GetOpPluginHandle();
  if (handle == nullptr) {
    MS_LOG(ERROR) << "Op plugin handle is not initialized. Please ensure MS_OP_PLUGIN_PATH is set correctly.";
    return -1;
  }

  // Clear previous errors before dlsym
  (void)DL_ERROR();
#ifdef _WIN32
  SetLastError(0);
#endif
  op_plugin_func =
    reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
      DL_SYM(handle, op_name.c_str()));
  if (auto error_info = DL_ERROR(); error_info != nullptr) {
    MS_LOG(ERROR) << "Failed to load op plugin kernel function for '" << op_name << "'. Error info: " << error_info;
    return -1;
  }

  return op_plugin_func(nparam, params, ndims, shapes, dtypes, stream, kernel_info);
}

int LaunchOpPluginKernel(const std::string &op_name, OpPluginKernelParam *param) {
  return LaunchOpPluginKernel(op_name, param->params.size(), param->params.data(), param->ndims.data(),
                              param->shapes.data(), param->dtypes.data(), reinterpret_cast<void *>(&param->kernel_info),
                              param->stream);
}
}  // namespace op_plugin
}  // namespace mindspore::kernel
