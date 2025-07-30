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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
#include <vector>
#include <memory>
#include <type_traits>
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/device/cpu/kernel/custom/op_plugin_utils.h"

// Helper to check if a type is optional
template <typename T>
struct is_std_optional : std::false_type {};

template <typename U>
struct is_std_optional<std::optional<U>> : std::true_type {};

template <typename T>
constexpr bool is_std_optional_v = is_std_optional<std::decay_t<T>>::value;

// Helper to check if a type is int or vector<int>
template <typename T>
struct is_int_or_vector_int : std::false_type {};

template <>
struct is_int_or_vector_int<int64_t> : std::true_type {};

template <>
struct is_int_or_vector_int<std::vector<int64_t>> : std::true_type {};

template <typename T>
constexpr bool is_int_or_vector_int_v = is_int_or_vector_int<std::decay_t<T>>::value;

template <typename... Args>
constexpr bool has_int_or_vector_int_v = (is_int_or_vector_int_v<Args> || ...);

namespace mindspore::kernel::pyboost {
template <typename T>
constexpr bool is_tensor_ptr_v = std::is_same_v<std::decay_t<T>, tensor::TensorPtr>;

struct InplaceInfo {
  bool all_outputs_inplace{false};
  std::vector<size_t> non_inplace_outputs;
};

InplaceInfo GetInplaceInfo(const std::string& op_name) {
  InplaceInfo info;
  auto op_def = GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(EXCEPTION) << "OpDef for " << op_name << " is not found.";
  }
  for (const auto &output : op_def->returns_) {
    if (output.inplace_input_index_ == -1) {
      info.non_inplace_outputs.push_back(&output - &op_def->returns_[0]);
    }
  }
  info.all_outputs_inplace = info.non_inplace_outputs.empty();
  return info;
}

// Overload for when any argument is int or vector<int> - returns empty vector
// Reason to have this overload:
// Some pyboost functions pass int or vector<int> as arguments, which are not compatible with the InferOutput function.
// These functions are mainly view functions, which do not really have an op plugin kernel.
template <typename... Args>
std::enable_if_t<has_int_or_vector_int_v<Args...>, std::vector<tensor::TensorPtr>> PyboostLaunchOpPluginKernel(
  std::shared_ptr<OpRunner> op, Args &&... args) {
  return {};
}

template <typename... Args>
std::enable_if_t<!has_int_or_vector_int_v<Args...>, std::vector<tensor::TensorPtr>> PyboostLaunchOpPluginKernel(
  std::shared_ptr<OpRunner> op, Args &&... args) {
  MS_EXCEPTION_IF_NULL(op->primitive());
  const auto &op_name = op->primitive()->name();
  MS_LOG(DEBUG) << op_name << " calls op plugin kernel.";

  const auto &inplace_info = GetInplaceInfo(op_name);

  if (!inplace_info.all_outputs_inplace) {
    op->InferOutput(args...);
  }

  const auto device_context = op->device_context();
  MS_EXCEPTION_IF_NULL(device_context);

  // Find tensor arguments for PrepareOpInputs
  auto process_tensor_args = [&](auto &&arg) {
    if constexpr (is_std_optional_v<decltype(arg)>) {
      if constexpr (is_tensor_ptr_v<decltype(arg.value())>) {
        PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), arg);
      }
    } else if constexpr (is_tensor_ptr_v<decltype(arg)>) {
      PyBoostUtils::PrepareOpInputs(device_context, op->stream_id(), arg);
    }
  };
  (process_tensor_args(args), ...);

  // Create device address for output tensors
  const auto &outputs = op->outputs();
  std::vector<tensor::TensorPtr> non_inplace_outputs;
  if (!inplace_info.all_outputs_inplace) {
    non_inplace_outputs.reserve(inplace_info.non_inplace_outputs.size());
    for (const auto &idx : inplace_info.non_inplace_outputs) {
      if (idx >= outputs.size()) {
        MS_LOG(EXCEPTION) << "Index " << idx << " is out of bounds for outputs of size " << outputs.size();
      }
      non_inplace_outputs.push_back(outputs[idx]);
    }
  } else {
    non_inplace_outputs = outputs;
  }

  // get non-inplace outputs
  PyBoostUtils::PrepareOpOutputs(device_context, 0, non_inplace_outputs);

  op->ProfileTrackerTask();

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, &op_name, args..., non_inplace_outputs]() {
    auto device_context = op->device_context();

    // Process tensor arguments for MallocOpInputs
    auto malloc_tensor_args = [&](auto &&arg) {
      if constexpr (is_std_optional_v<decltype(arg)>) {
        if constexpr (is_tensor_ptr_v<decltype(arg.value())>) {
          PyBoostUtils::MallocOpInputs(device_context, arg);
        }
      } else if constexpr (is_tensor_ptr_v<decltype(arg)>) {
        PyBoostUtils::MallocOpInputs(device_context, arg);
      }
    };
    (malloc_tensor_args(args), ...);
    PyBoostUtils::MallocOpOutputs(device_context, non_inplace_outputs);

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), args...);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, non_inplace_outputs);
    std::vector<kernel::KernelTensor *> workspace_tensors;
    auto op_plugin_param = CreateOpPluginParam(input_address_info.first, output_address_info.first, workspace_tensors);
    auto ret = LaunchOpPluginKernel(op_name, &op_plugin_param);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "Launch op plugin kernel failed, op name: " << op_name << ", return code: " << ret;
    }
  }));
  op->ProfileTrackerInput(args...);
  op->ProfileTrackerOutput(outputs);
  MS_LOG(DEBUG) << op_name << " op plugin kernel call end";
  op->CreateOutputSimpleInfo();
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_PYBOOST_OP_PLUGIN_UTILS_H_
