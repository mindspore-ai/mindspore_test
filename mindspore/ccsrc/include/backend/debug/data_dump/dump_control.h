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
#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_CONTROL_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DUMP_CONTROL_H_

#include <string>
#include <mutex>
#include <vector>
#include <memory>
#include <regex>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/visible.h"

namespace mindspore {

class BACKEND_EXPORT DumpControl {
 public:
  static DumpControl &GetInstance() {
    std::call_once(dump_mutex_, []() {
      if (dump_instance_ == nullptr) {
        dump_instance_ = std::shared_ptr<DumpControl>(new DumpControl);
      }
    });
    return *dump_instance_;
  }
  ~DumpControl() = default;
  bool dynamic_switch() const { return dynamic_switch_; }
  bool dump_switch() const { return dump_switch_; }

  void SetDynamicDump() { dynamic_switch_ = true; }
  void DynamicDumpStart();
  void DynamicDumpStop();

 private:
  DumpControl() = default;
  DISABLE_COPY_AND_ASSIGN(DumpControl)
  inline static std::shared_ptr<DumpControl> dump_instance_ = nullptr;
  inline static std::once_flag dump_mutex_;
  bool dynamic_switch_{false};
  bool dump_switch_{false};
};

}  // namespace mindspore

#endif
