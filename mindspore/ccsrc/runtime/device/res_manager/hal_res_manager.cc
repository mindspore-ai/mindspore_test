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
#include "runtime/device/res_manager/hal_res_manager.h"

#include <memory>

#include "utils/ms_context.h"
#ifndef _WIN32
#include <libgen.h>
#endif
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace device {
namespace {
const char *kAscendResManagerName = "libmindspore_ascend_res_manager.so";
const char *kGPUResManagerName = "libmindspore_gpu_res_manager.so";
const char *kCPUResManagerName = "libmindspore_cpu_res_manager.so";
const std::map<DeviceType, std::string> device_type_to_lib_name = {{DeviceType::kAscend, kAscendResManagerName},
                                                                   {DeviceType::kGPU, kGPUResManagerName},
                                                                   {DeviceType::kCPU, kCPUResManagerName}};
const std::map<DeviceType, std::string> device_type_to_plugin_name = {
  {DeviceType::kAscend, "ascend"}, {DeviceType::kGPU, "gpu"}, {DeviceType::kCPU, "cpu"}};

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
  HMODULE hModule = nullptr;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                        (LPCSTR)GetCurrentDir, &hModule) == 0) {
    MS_LOG(WARNING) << "Get GetModuleHandleEx failed.";
    return "";
  }
  char szPath[MAX_PATH];
  if (GetModuleFileName(hModule, szPath, sizeof(szPath)) == 0) {
    MS_LOG(WARNING) << "Get GetModuleFileName failed.";
    return "";
  }
  std::string cur_so_path = std::string(szPath);
  auto pos = cur_so_path.find_last_of("\\");
  if (cur_so_path.empty() || pos == std::string::npos) {
    MS_LOG(ERROR) << "Current so path empty or the path [" << cur_so_path << "] is invalid.";
    return "";
  }
  return cur_so_path.substr(0, pos);
#endif
}

std::string GetHalResManagerLibName(const DeviceType &device_name) {
  auto iter = device_type_to_lib_name.find(device_name);
  if (iter != device_type_to_lib_name.end()) {
    return iter->second;
  }
  return "";
}

std::string GetPluginName(const DeviceType &device_name) {
  auto iter = device_type_to_plugin_name.find(device_name);
  if (iter != device_type_to_plugin_name.end()) {
    return iter->second;
  }
  MS_LOG(EXCEPTION) << "Can not find plugin name for device: " << device_name;
  return "";
}
}  // namespace

HalResManager &HalResManager::GetInstance() {
  static HalResManager instance{};
  return instance;
}

void HalResManager::Clear() { multi_stream_controllers_.clear(); }

void HalResManager::Register(const DeviceType device, HalResManagerCreator &&hal_res_manager_creator) {
  if (hal_res_manager_creators_.find(device) == hal_res_manager_creators_.end()) {
    (void)hal_res_manager_creators_.emplace(device, hal_res_manager_creator);
  }
}

void HalResManager::LoadResManager(const DeviceType &device_name) {
  if (loaded_res_manager_handles_.count(device_name) != 0) {
    return;
  }

  auto device_lib_name = GetHalResManagerLibName(device_name);
  MS_LOG(INFO) << "HalResManager dlopen ascend lib name: " << device_lib_name;
  void *handle;
  std::string err_msg = "";
  std::string cur_lib_name = "";
#ifdef _WIN32
  if (device_name != DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Only the CPU res manager support dynamic load in Windows.";
  }
  cur_lib_name = GetCurrentDir() + "\\mindspore_cpu_res_manager.dll";
  MS_LOG(INFO) << "HalResManager dynamic load " << cur_lib_name;
  handle = LoadLibraryEx(cur_lib_name.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  err_msg = std::to_string(GetLastError());
#elif defined(__APPLE__)
  if (device_name != DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Only the CPU res manager support dynamic load in Mac.";
  }
  cur_lib_name = GetCurrentDir() + "/plugin/" + GetPluginName(device_name) + "/" + "libmindspore_cpu_res_manager.dylib";
  MS_LOG(INFO) << "HalResManager dlopen current device lib name: " << cur_lib_name;
  handle = dlopen(cur_lib_name.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#else
  cur_lib_name = GetCurrentDir() + "/plugin/" + GetPluginName(device_name) + "/" + device_lib_name;
  MS_LOG(INFO) << "HalResManager dlopen current device lib name: " << cur_lib_name;
  handle = dlopen(cur_lib_name.c_str(), RTLD_LAZY);
  err_msg = GetDlErrorMsg();
#endif

  if (handle == nullptr) {
    MS_LOG(EXCEPTION) << "Loading " + cur_lib_name + " failed. Error: " + err_msg;
  }
  (void)loaded_res_manager_handles_.emplace(device_name, handle);
}

void HalResManager::UnLoadResManager(const DeviceType &device_name) {
  for (const auto &iter : loaded_res_manager_handles_) {
    auto device_lib_name = GetHalResManagerLibName(iter.first);
    auto handle = iter.second;
#ifndef _WIN32
    if (dlclose(handle) != 0) {
      MS_LOG(EXCEPTION) << "Closing " + device_lib_name + " handle failed. Error: " + GetDlErrorMsg();
    }
#else
    if (!FreeLibrary(reinterpret_cast<HINSTANCE__ *>(handle))) {
      MS_LOG(EXCEPTION) << "Closing " + device_lib_name + " handle failed. Error: " + std::to_string(GetLastError());
    }
#endif
  }
}

HalResBase *HalResManager::GetOrCreateResManager(const ResKey &res_key) {
  auto res_manager_iter = res_managers_.find(res_key.ToString());
  if (res_manager_iter != res_managers_.end()) {
    return res_manager_iter->second.get();
  }

#ifdef WITH_BACKEND
  // dynamic load res_manager library.
  if (res_key.device_name_ == DeviceType::kAscend || res_key.device_name_ == DeviceType::kGPU) {
    LoadResManager(res_key.device_name_);
  }
#endif

  std::shared_ptr<HalResBase> res_manager;
  auto creator_iter = hal_res_manager_creators_.find(res_key.device_name_);
  if (creator_iter != hal_res_manager_creators_.end()) {
    res_manager = (creator_iter->second)(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    res_managers_[res_key.ToString()] = res_manager;
    multi_stream_controllers_[res_key.DeviceName()] = std::make_shared<MultiStreamController>(res_manager.get());
  } else {
    MS_LOG(EXCEPTION) << "Create resource manager failed, please make sure target device:" << res_key.ToString()
                      << " is valid.";
  }
  return res_manager.get();
}

HalResPtr HalResManager::GetResManager(const ResKey &res_key) {
  if (res_managers_.count(res_key.ToString()) == 0) {
    MS_LOG(INFO) << "ResManager of device " << res_key.ToString() << " is not created yet.";
    return nullptr;
  }
  return res_managers_[res_key.ToString()];
}

MultiStreamControllerPtr &HalResManager::GetMultiStreamController(const std::string &device_name) {
  auto &&iter = multi_stream_controllers_.find(device_name);
  if (iter != multi_stream_controllers_.end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "Found multi stream controller failed, and try to initialize, device_name : " << device_name
                  << ".";
  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto res_key = ResKey{GetDeviceTypeByName(device_name), device_id};
  auto hal_res_base = GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(hal_res_base);
  auto &&iter_again = multi_stream_controllers_.find(device_name);
  if (iter_again == multi_stream_controllers_.end()) {
    MS_LOG(EXCEPTION) << "Get multi stream controller failed, device_name : " << device_name << ".";
  }
  return iter_again->second;
}
}  // namespace device
}  // namespace mindspore
