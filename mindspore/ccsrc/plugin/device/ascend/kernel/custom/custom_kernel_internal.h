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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_INTERNAL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_INTERNAL_H_

#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"
#include <vector>

namespace mindspore {
namespace kernel {

// Internal macros for hardware format mapping registration
// These macros are for internal use only and should not be exposed in public headers

#define MS_CUSTOM_KERNEL_HARDWARE_FORMAT_MAPPING_REG(NAME, HARDWARE, ...)                              \
  static const bool g_##NAME##_##HARDWARE##_format_mapping_registered __attribute__((unused)) = []() { \
    auto &factory = mindspore::kernel::CustomKernelFactory::Instance();                                \
    std::vector<mindspore::kernel::KernelFormatMapping> format_mappings = {__VA_ARGS__};               \
    mindspore::kernel::HardwareFormatMapping hardware_mapping(#HARDWARE, format_mappings);             \
    factory.RegisterHardwareFormatMapping("Custom_" #NAME, hardware_mapping);                          \
    return true;                                                                                       \
  }();

#define MS_FORMAT_MAPPING(INPUT_FORMATS, OUTPUT_FORMATS) \
  mindspore::kernel::KernelFormatMapping(INPUT_FORMATS, OUTPUT_FORMATS)

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_INTERNAL_H_
