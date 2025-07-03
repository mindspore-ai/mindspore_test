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

#include "utils/distributed_meta.h"
#include <mutex>
#include "utils/log_adapter.h"

namespace mindspore {
std::shared_ptr<DistributedMeta> DistributedMeta::instance_ = nullptr;

std::shared_ptr<DistributedMeta> DistributedMeta::GetInstance() {
  static std::once_flag instance_init_flag_ = {};
  std::call_once(instance_init_flag_, [&]() {
    if (instance_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore DistributedMeta";
      instance_ = std::make_shared<DistributedMeta>();
    }
  });
  MS_EXCEPTION_IF_NULL(instance_);
  return instance_;
}
}  // namespace mindspore
