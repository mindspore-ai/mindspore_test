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

#ifndef MINDSPORE_CCSRC_TOOLS_SILENT_CHECK_SILENT_DETECTOR_H
#define MINDSPORE_CCSRC_TOOLS_SILENT_CHECK_SILENT_DETECTOR_H

#include <deque>
#include <optional>
#include <unordered_map>
#include <string>
#include "include/common/visible.h"
#include "ir/tensor.h"

namespace mindspore {
namespace silentdetect {

DUMP_EXPORT void SilentDetect(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);

struct StatData {
  double avg = 0.0;
  double pre_value = 0.0;
  int count = 0;
  int none_zero_count = 0;
};

struct StrikeRecord {
  std::chrono::system_clock::time_point timestamp;
  std::string name;
  double value;
  StatData stat;
};

class DUMP_EXPORT SilentDetector {
 public:
  static SilentDetector &GetInstance() {
    static SilentDetector instance;
    return instance;
  }

  SilentDetector(const SilentDetector &) = delete;
  SilentDetector &operator=(const SilentDetector &) = delete;
  SilentDetector(SilentDetector &&) = delete;
  SilentDetector &operator=(SilentDetector &&) = delete;
  std::optional<StrikeRecord> CheckValue(const string &name, double value);
  std::optional<StrikeRecord> CheckValueWithCoolDown(const string &name, double value, int cooldown);
  friend void SilentDetect(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);

 private:
  SilentDetector();
  std::unordered_map<std::string, StatData> check_status_;
  std::chrono::system_clock::time_point earliest_strike_time_;
};

}  // namespace silentdetect
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TOOLS_SILENT_CHECK_SILENT_DETECTOR_H
