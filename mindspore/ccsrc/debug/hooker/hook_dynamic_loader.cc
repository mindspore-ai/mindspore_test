/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "debug/hooker/hook_dynamic_loader.h"
#include "utils/log_adapter.h"

HookDynamicLoader &HookDynamicLoader::GetInstance() {
  static HookDynamicLoader instance;
  return instance;
}

bool HookDynamicLoader::loadFunction(void *handle, const std::string &functionName) {
  void *func = dlsym(handle, functionName.c_str());
  if (!func) {
    MS_LOG(WARNING) << "Could not load function: " << functionName;
    return false;
  }
  funcMap_[functionName] = func;
  return true;
}

bool HookDynamicLoader::LoadLibrary() {
  const char *libPath = std::getenv("HOOK_TOOL_PATH");
  if (!libPath) {
    MS_LOG(WARNING) << "HOOK_TOOL_PATH is not set!";
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (handle_) {
    MS_LOG(WARNING) << "Hook library already loaded!";
    return false;
  }
  handle_ = dlopen(libPath, RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    MS_LOG(WARNING) << "Failed to load Hook library";
    return false;
  }

  for (const std::string &functionName : functionList_) {
    if (!loadFunction(handle_, functionName)) {
      MS_LOG(WARNING) << "Failed to load so function";
      dlclose(handle_);
      return false;
    }
  }
  MS_LOG(INFO) << "Load Hook library success.";
  return true;
}

bool HookDynamicLoader::UnloadLibrary() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!handle_) {
    MS_LOG(WARNING) << "Hook library hasn't been loaded";
    return false;
  }
  dlclose(handle_);
  handle_ = nullptr;
  funcMap_.clear();
  return true;
}

void *HookDynamicLoader::GetHooker(std::string funcName) {
  auto iter = funcMap_.find(funcName);
  if (iter == funcMap_.end()) {
    return nullptr;
  }
  return iter->second;
}
