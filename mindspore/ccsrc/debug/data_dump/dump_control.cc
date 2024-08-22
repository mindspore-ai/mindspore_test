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

#include "include/backend/debug/data_dump/dump_control.h"
#include "utils/log_adapter.h"

namespace mindspore {

void DumpControl::DynamicDumpStart() {
  if (!dynamic_switch_) {
    MS_LOG(WARNING) << "dump_start before dump_set_dynamic-Warning: dump_set_dynamic has not been set!";
  }
  dump_switch_ = true;
}

void DumpControl::DynamicDumpStop() {
  if (!dynamic_switch_) {
    MS_LOG(WARNING) << "dump_stop before dump_set_dynamic-Warning: dump_set_dynamic has not been set!";
  }
  dump_switch_ = false;
}
}  // namespace mindspore
