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
#include "runtime/device/res_manager/hal_res_base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
void *HalResBase::AllocateOffloadMemory(size_t size) const {
  MS_LOG(EXCEPTION) << "Not implemented interface.";
  return nullptr;
}

void HalResBase::FreeOffloadMemory(void *ptr) const {
  MS_LOG(EXCEPTION) << "Not implemented interface.";
  return;
}
}  // namespace device
}  // namespace mindspore
