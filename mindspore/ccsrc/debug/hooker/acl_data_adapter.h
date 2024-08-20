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

#ifndef MINDSPORE_CCSRC_DEBUG_HOOKER_ACL_DATA_ADAPTER_H_
#define MINDSPORE_CCSRC_DEBUG_HOOKER_ACL_DATA_ADAPTER_H_

#include <string>
#include <vector>
#include <map>
#include "debug/hooker/adapter.h"
#include "debug/hooker/hook_dynamic_loader.h"

#define TO_MAP(var, id, map) map[id] = static_cast<void *>(&var);

using HookBeginPtr = void (*)(uint32_t device_id, int step_count_num, std::map<uint32_t, void *> ext_args);
using HookEndPtr = void (*)(std::map<uint32_t, void *> ext_args);

namespace mindspore {
namespace hooker {
class AclDataAdapter : public Adapter {
 public:
  void AdaptOnStepBegin(uint32_t device_id, int step_count_num, std::vector<std::string> all_kernel_names,
                        bool is_kbyk) override;

  void AdaptOnStepEnd() override;

  AclDataAdapter();

  ~AclDataAdapter() {}

 private:
  bool isLoaded = false;
};

}  // namespace hooker
}  // namespace mindspore
#endif
