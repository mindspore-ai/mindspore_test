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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_DISPATCH_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_DISPATCH_H_

#include "runtime/utils/visible.h"
#include "ir/tensor.h"
#include "ir/anf.h"
#include "pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "include/runtime/utils/dispatch/dispatch_env.h"

namespace mindspore {
constexpr int kDeviceTypeCount = static_cast<int>(device::DeviceType::kDeviceEnd);
static_assert(kDeviceTypeCount <= 16, "DeviceType count must be <= 16.");

template <typename T>
struct is_tensor : std::false_type {};

template <>
struct is_tensor<mindspore::tensor::TensorPtr> : std::true_type {};

template <typename T>
struct is_value_tuple : std::false_type {};

template <>
struct is_value_tuple<ValueTuplePtr> : std::true_type {};

std::string GetPythonStackTrace();

template <typename T>
device::DeviceType get_device_single(const T &input) {
  if constexpr (is_tensor<T>::value) {
    MS_LOG(DEBUG) << "[Dispatch]Get tensor device " << device::GetDeviceNameByType(input->device_type());
    return input->device_type();
  } else if constexpr (is_value_tuple<T>::value) {
    MS_LOG(DEBUG) << "[Dispatch]Get device from ValueTuple.";
    auto result = device::DeviceType::kUnknown;
    const auto &values = input->value();
    for (const auto &t : values) {
      if (t->template isa<mindspore::tensor::Tensor>()) {
        auto tensor = t->template cast<mindspore::tensor::TensorPtr>();
        if (tensor->device_type() > result) {
          result = tensor->device_type();
          MS_LOG(DEBUG) << "[Dispatch]Get tensor device " << device::GetDeviceNameByType(result);
        }
      }
    }
    return result;
  } else {
    MS_LOG(DEBUG) << "[Dispatch] Unknown input type.";
    return device::DeviceType::kUnknown;
  }
}

inline void CheckDeviceCount(uint32_t device_mask) {
  int device_count = 0;
  for (int i = 0; i < kDeviceTypeCount; ++i) {
    if (device_mask & (uint32_t(1u) << i)) {
      ++device_count;
    }
  }

  if (device_count > 1) {
    std::ostringstream oss;
    oss << "[Dispatch]Found multiple device types in inputs: ";
    bool first = true;
    for (int i = 0; i < kDeviceTypeCount; ++i) {
      if (!(device_mask & (uint32_t(1u) << i))) {
        continue;
      }
      if (!first) {
        oss << ", ";
      }
      first = false;
      auto d = static_cast<device::DeviceType>(i);
      oss << device::GetDeviceNameByType(d);
    }
    oss << ".";
    if (EnableDispatchWithStack()) {
      oss << " Stack:\n" << GetPythonStackTrace();
    }
    MS_LOG(WARNING) << oss.str();
  }
}

inline int GetHighPriorityDevice(uint32_t device_mask) {
  int highest_idx = -1;
  for (int i = kDeviceTypeCount - 1; i >= 0; --i) {
    if (device_mask & (uint32_t(1u) << i)) {
      highest_idx = i;
      break;
    }
  }
  return highest_idx;
}

template <typename... Args>
device::DeviceType get_device(const Args &...args) {
  uint32_t device_mask = 0;

  auto collect = [&](device::DeviceType d) {
    if (d == device::DeviceType::kUnknown || d == device::DeviceType::kNone) {
      return;
    }
    int idx = static_cast<int>(d);
    device_mask |= (uint32_t(1u) << idx);
  };

  (collect(get_device_single(args)), ...);

  if (device_mask == 0) {
    auto result = kernel::pyboost::OpRunStatus::Get().device_target();
    MS_LOG(WARNING) << "[Dispatch]No input device found. Use default device " << device::GetDeviceNameByType(result);
    return result;
  }

  CheckDeviceCount(device_mask);

  int highest_idx = GetHighPriorityDevice(device_mask);
  if (highest_idx < 0) {
    auto result = kernel::pyboost::OpRunStatus::Get().device_target();
    MS_LOG(WARNING) << "[Dispatch]Device mask is invalid. Use default device " << device::GetDeviceNameByType(result);
    return result;
  }

  auto result = static_cast<device::DeviceType>(highest_idx);
  MS_LOG(DEBUG) << "[Dispatch]Dispatch device " << device::GetDeviceNameByType(result);
  return result;
}
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_PYBOOST_FUNCTIONS_DISPATCH_H_
