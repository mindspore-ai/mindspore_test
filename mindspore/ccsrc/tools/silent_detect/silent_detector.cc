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

#include "tools/silent_detect/silent_detector.h"

#include <string>

#include "tools/silent_detect/silent_detect_config_parser.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace silentdetect {
namespace {
std::optional<double> ScalarTensorToDouble(mindspore::tensor::TensorPtr tensor, const std::string &name) {
  MS_EXCEPTION_IF_NULL(tensor);
  size_t size = SizeOf(tensor->shape());
  MS_EXCEPTION_IF_CHECK_FAIL(size == 1, "For silent detect feature value, there must be only one element, but got " +
                                          std::to_string(size) + ".");
  auto cpu_tensor = tensor->cpu();
  auto data_type = cpu_tensor->data_type();
  auto data = cpu_tensor->data_c();
  switch (data_type) {
    case TypeId::kNumberTypeBFloat16:
      return static_cast<double>(*static_cast<const bfloat16 *>(data));
    case TypeId::kNumberTypeFloat16:
      return static_cast<double>(*static_cast<const float16 *>(data));
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
      return static_cast<double>(*static_cast<const float *>(data));
    case TypeId::kNumberTypeDouble:
    case TypeId::kNumberTypeFloat64:
      return *static_cast<const double *>(data);
    default:
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Skip the unsupported data type: " << TypeIdToString(data_type)
                                      << ", tensor name is " << name;
      return std::nullopt;
  }
}

std::string ToString(const StatData &data) {
  std::stringstream ss;
  ss << "StatData{"
     << "avg: " << data.avg << ", pre_value: " << data.pre_value << ", count: " << data.count
     << ", none_zero_count: " << data.none_zero_count << "}";
  return ss.str();
}

std::string ToString(const StrikeRecord &record) {
  std::stringstream ss;
  ss << "StrikeRecord{"
     << "timestamp: " << std::chrono::system_clock::to_time_t(record.timestamp) << ", name: " << record.name
     << ", value: " << record.value << ", stat: " << ToString(record.stat) << "}";
  return ss.str();
}

}  // namespace

void SilentDetect(std::string name, mindspore::tensor::TensorPtr tensor) {
  auto res = ScalarTensorToDouble(tensor, name);
  if (res.has_value()) {
    auto current_val = res.value();
    auto cool_down = SilentDetectConfigParser::GetInstance().GetCooldown();
    auto strike_record = SilentDetector::GetInstance().CheckValueWithCoolDown(name, current_val, cool_down);
    // todo: handle the strike record, start checksum.
    if (strike_record.has_value()) {
      MS_LOG(WARNING) << "Silent detect strike detected: " << ToString(strike_record.value());
    }
  }
}

SilentDetector::SilentDetector() { earliest_strike_time_ = std::chrono::system_clock::time_point::min(); }

std::optional<StrikeRecord> SilentDetector::CheckValueWithCoolDown(const std::string &name, double value,
                                                                   int cooldown) {
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Silent detect receives data, name is " << name << ", value is " << value;
  auto &stat = check_status_[name];
  auto strike_record = CheckValue(name, value);
  stat.pre_value = value;
  stat.count += 1;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "After silent detect, name is " << name << ", stat is " << ToString(stat);
  if (!strike_record.has_value()) {
    return std::nullopt;
  }
  auto strike_time = strike_record.value().timestamp;
  if (earliest_strike_time_ + std::chrono::minutes(cooldown) < strike_time) {
    earliest_strike_time_ = strike_time;
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Strike happened. record is " << ToString(strike_record.value());
    return strike_record;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Strike will not be recorded repeatedly during the " << cooldown
                                  << " mins cooling-off period, and the exception info is "
                                  << ToString(strike_record.value());
  return std::nullopt;
}

std::optional<StrikeRecord> SilentDetector::CheckValue(const std::string &name, double value) {
  static const int alpha1 = SilentDetectConfigParser::GetInstance().GetUpperThresh1();
  static const int alpha2 = SilentDetectConfigParser::GetInstance().GetUpperThresh2();

  auto &stat = check_status_[name];

  if (value == 0.0) {
    return std::nullopt;
  }

  if (std::isnan(value) || std::isinf(value)) {
    StrikeRecord record{std::chrono::system_clock::now(), name, value, stat};
    return record;
  }

  double thres = value;
  double thres2 = value;
  if (stat.none_zero_count >= 10 && stat.avg != 0.0) {
    const double factor = 1.0 - std::pow(0.99, stat.none_zero_count);
    thres = stat.avg * static_cast<double>(alpha1) / factor;
    thres2 = stat.avg * static_cast<double>(alpha2) / factor;
  }

  if (value > thres && std::abs(value - stat.pre_value) > thres) {
    StrikeRecord record{std::chrono::system_clock::now(), name, value, stat};
    return record;
  }

  if (value <= thres2) {
    stat.none_zero_count += 1;
    stat.avg = stat.avg * 0.99 + value * 0.01;
  }
  return std::nullopt;
}

}  // namespace silentdetect
}  // namespace mindspore
