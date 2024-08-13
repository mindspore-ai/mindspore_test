/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/internal_kernel_build.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "plugin/device/ascend/kernel/internal/kernel_plugin.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
static std::shared_ptr<KernelPlugin> k_internal_kernel_plugin_ptr = nullptr;
static bool k_is_plugin_init = false;
std::shared_ptr<KernelPlugin> GetKernelPLugin() {
  if (k_is_plugin_init) {
    return k_internal_kernel_plugin_ptr;
  }

  // tryp load so file
  std::string cur_so_path;
  auto ret = plugin_loader::PluginLoader::GetPluginPath(&cur_so_path);
  MS_LOG(INFO) << "get so path " << cur_so_path << " return " << ret;
  auto targe_so_path = cur_so_path + "/ascend/" + "libmindspore_internal_kernels.so";

  std::stringstream dlopen_error_msg;
  std::map<std::string, void *> plugin_maps;
  ret = plugin_loader::PluginLoader::LoadDynamicLib(targe_so_path, &plugin_maps, &dlopen_error_msg);
  if (!ret) {
    MS_LOG(ERROR) << "load so failed " << dlopen_error_msg.str();
  }

  // create plugin object
  k_internal_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("InternalKernelPlugin");
  k_is_plugin_init = true;
  return k_internal_kernel_plugin_ptr;
}

KernelModPtr InternalKernelBuild(const AnfNodePtr &anf_node) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return internal_kernel_plugin_ptr->BuildKernel(anf_node);
}

bool IsRegisteredInternalKernel(const AnfNodePtr &anf_node) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return internal_kernel_plugin_ptr->IsRegisteredKernel(anf_node);
}

void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node, std::vector<std::string> *input_formats,
                                               std::vector<std::string> *output_formats) {
  auto internal_kernel_plugin_ptr = GetKernelPLugin();
  if (internal_kernel_plugin_ptr == nullptr) {
    return;
  }
  return internal_kernel_plugin_ptr->GetValidKernelBuildInfoWithInternalFormat(node, input_formats, output_formats);
}
}  // namespace kernel
}  // namespace mindspore
