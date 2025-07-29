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

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <string>
#include <algorithm>
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"
#include "plugin/device/cpu/kernel/custom/op_plugin_utils.h"

namespace mindspore::kernel {
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
#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryA(real_path.c_str());
  if (handle == nullptr) {
    DWORD error = GetLastError();
    MS_LOG(WARNING) << "Failed to open op plugin file: " << real_path << " Error code: " << error;
  }
#else
  handle = dlopen(real_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(WARNING) << "Failed to open op plugin file: " << dlerror();
  }
#endif

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
    const std::string reg_func_name = "IsKernelRegistered";
#if defined(_WIN32) || defined(_WIN64)
    reg_func = reinterpret_cast<std::add_pointer<bool(const char *)>::type>(
      GetProcAddress(static_cast<HMODULE>(handle), reg_func_name.c_str()));
    if (reg_func == nullptr) {
      DWORD error = GetLastError();
      MS_LOG(WARNING) << "Error occurs when fetching function '" << reg_func_name
                      << "' from op plugin library. Error code: " << error;
      return false;
    }
#else
    reg_func = reinterpret_cast<std::add_pointer<bool(const char *)>::type>(dlsym(handle, reg_func_name.c_str()));
    if (auto error_info = dlerror(); error_info != nullptr) {
      MS_LOG(WARNING) << "Error occurs when fetching function '" << reg_func_name
                      << "' from libmindspore_op_plugin.so. Error info: " << error_info;
      return false;
    }
#endif
  }
  return reg_func != nullptr && reg_func(op_name.c_str());
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
#if defined(_WIN32) || defined(_WIN64)
  op_plugin_func =
    reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
      GetProcAddress(static_cast<HMODULE>(handle), op_name.c_str()));
  if (op_plugin_func == nullptr) {
    DWORD error = GetLastError();
    MS_LOG(ERROR) << "Failed to load op plugin kernel function for '" << op_name << "'. Error code: " << error;
    return -1;
  }
#else
  // Clear previous errors before dlsym
  dlerror();
  op_plugin_func =
    reinterpret_cast<std::add_pointer<int(int, void **, int *, int64_t **, const char **, void *, void *)>::type>(
      dlsym(handle, op_name.c_str()));
  if (auto error_info = dlerror(); error_info != nullptr) {
    MS_LOG(ERROR) << "Failed to load op plugin kernel function for '" << op_name << "'. Error info: " << error_info;
    return -1;
  }
#endif

  return op_plugin_func(nparam, params, ndims, shapes, dtypes, kernel_info, stream);
}

int LaunchOpPluginKernel(const std::string &op_name, OpPluginKernelParam *param) {
  return LaunchOpPluginKernel(op_name, param->params.size(), param->params.data(), param->ndims.data(),
                              param->shapes.data(), param->dtypes.data(), reinterpret_cast<void *>(&param->kernel_info),
                              param->stream);
}
}  // namespace mindspore::kernel
