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
#include "include/backend/distributed/collective/collect_hccl_init_info.h"
#include <mutex>
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
namespace collective {
std::shared_ptr<CollectHcclInitInfo> CollectHcclInitInfo::GetInstance() {
  static std::once_flag init_flag;
  static std::shared_ptr<CollectHcclInitInfo> instance = nullptr;
  std::call_once(init_flag, [&]() {
    if (instance == nullptr) {
      instance.reset(new (std::nothrow) CollectHcclInitInfo());
      MS_EXCEPTION_IF_NULL(instance);
    }
  });
  return instance;
}
uint32_t CollectHcclInitInfo::GetBuffsize(const std::string &group_name) {
  uint32_t buffsize = 0;
  auto iter = group_with_buffsize_.find(group_name);
  if (iter != group_with_buffsize_.end()) {
    buffsize = iter->second;
  }
  return buffsize;
}
void *CollectHcclInitInfo::GetRootInfo(const std::string &group_name) {
  void *root_info = nullptr;
  auto iter = group_with_root_info_.find(group_name);
  if (iter != group_with_root_info_.end()) {
    root_info = iter->second;
  }
  return root_info;
}
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
