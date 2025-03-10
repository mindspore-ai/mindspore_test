/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_ACL_DATA_ADAPTER_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_DUMP_ACL_DATA_ADAPTER_H_

#include <string>
#include <vector>
#include <map>
#include "backend/ge_backend/dump/adapter.h"
#include "backend/ge_backend/dump/hook_dynamic_loader.h"

namespace mindspore {
namespace dump {
using HookBeginPtr = void (*)(uint32_t device_id, int step_count_num, std::map<uint32_t, void *> ext_args);
using HookEndPtr = void (*)(std::map<uint32_t, void *> ext_args);

class AclDataAdapter : public Adapter {
 public:
  void AdaptOnStepBegin(uint32_t device_id, int step_count_num, std::vector<std::string> &&all_kernel_names,
                        bool is_kbyk) override;

  void AdaptOnStepEnd() override;

  void Load() override;

  AclDataAdapter() {}

  ~AclDataAdapter() {}

 private:
  bool isLoaded_ = false;
};

}  // namespace dump
}  // namespace mindspore
#endif
