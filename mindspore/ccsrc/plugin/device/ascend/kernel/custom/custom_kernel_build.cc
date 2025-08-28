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

#include "plugin/device/ascend/kernel/custom/custom_kernel_build.h"

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

#include "plugin/device/ascend/kernel/utils/kernel_plugin.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_internal.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
static std::shared_ptr<KernelPlugin> k_custom_kernel_plugin_ptr = nullptr;
static bool k_is_custom_plugin_init = false;
std::shared_ptr<KernelPlugin> GetCustomKernelPlugin() {
  if (k_is_custom_plugin_init) {
    return k_custom_kernel_plugin_ptr;
  }

  // create plugin object
  k_custom_kernel_plugin_ptr = Factory<KernelPlugin>::Instance().Create("CustomKernelPlugin");
  k_is_custom_plugin_init = true;

  return k_custom_kernel_plugin_ptr;
}

KernelModPtr CustomKernelBuild(const AnfNodePtr &anf_node) {
  auto custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (custom_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return custom_kernel_plugin_ptr->BuildKernel(anf_node);
}

bool IsRegisteredCustomKernel(const AnfNodePtr &anf_node) {
  auto custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (custom_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return custom_kernel_plugin_ptr->IsRegisteredKernel(anf_node);
}

bool IsEnableCustomNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return IsRegisteredCustomKernel(node);
}

static void KernelFormatRegister() {
  // reshape and cache
  MS_CUSTOM_KERNEL_HARDWARE_FORMAT_MAPPING_REG(
    reshape_and_cache, ascend310p,
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_FRAC_NZ,
                                                kOpFormat_FRAC_NZ, kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{})));

  MS_CUSTOM_KERNEL_HARDWARE_FORMAT_MAPPING_REG(
    reshape_and_cache, ascend910b,
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{})),
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT,
                                                kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{})),
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_FRAC_NZ,
                                                kOpFormat_FRAC_NZ, kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{})));
}

void ProcessCustomKernelFormatMapping(const CNodePtr &kernel, std::vector<std::string> *input_formats,
                                      std::vector<std::string> *output_formats) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(input_formats);
  MS_EXCEPTION_IF_NULL(output_formats);

  KernelFormatRegister();

  auto op_name = common::AnfAlgo::GetCNodeName(kernel);

  std::string hardware = "";
  auto context_ptr = MsContext::GetInstance();
  if (context_ptr != nullptr) {
    hardware = context_ptr->ascend_soc_version();
  }

  auto &factory = CustomKernelFactory::Instance();
  if (factory.HasFormatMapping(op_name)) {
    // Only get input formats when format mapping is needed
    for (size_t i = 0; i < input_formats->size(); i++) {
      auto prev_node = common::AnfAlgo::GetPrevNodeOutput(kernel, i);
      auto prev_format = AnfAlgo::GetOutputFormat(prev_node.first, prev_node.second);
      (*input_formats)[i] = prev_format;
    }

    // Find matching format mapping and check if found
    KernelFormatMapping format_mapping;
    bool found_matching = factory.FindMatchingFormatMapping(op_name, *input_formats, hardware, &format_mapping);

    if (found_matching) {
      // Apply output formats if any
      for (size_t i = 0; i < output_formats->size() && i < format_mapping.output_formats.size(); ++i) {
        if (!format_mapping.output_formats[i].empty()) {
          (*output_formats)[i] = format_mapping.output_formats[i];
          MS_LOG(INFO) << "Set output " << i << " format to: " << format_mapping.output_formats[i];
        }
      }

      if (!format_mapping.output_formats.empty()) {
        MS_LOG(INFO) << "Found matching format mapping for kernel: " << op_name
                     << " with input formats: " << *input_formats
                     << " and hardware: " << (hardware.empty() ? "unknown" : hardware);
      } else {
        MS_LOG(INFO) << "Found matching format mapping for inplace kernel: " << op_name
                     << " with input formats: " << *input_formats
                     << " and hardware: " << (hardware.empty() ? "unknown" : hardware)
                     << " (no output format mapping needed)";
      }
    } else {
      // No matching format mapping found, report error
      std::string input_formats_str = "[";
      for (size_t i = 0; i < input_formats->size(); ++i) {
        if (i > 0) input_formats_str += ", ";
        input_formats_str += (*input_formats)[i];
      }
      input_formats_str += "]";

      MS_EXCEPTION(ValueError) << "No matching format mapping found for custom kernel: " << op_name
                               << " with input formats: " << input_formats_str
                               << " and hardware: " << (hardware.empty() ? "unknown" : hardware)
                               << ". Please register appropriate format mapping.";
    }
  } else {
    MS_LOG(INFO) << "No format mapping found for custom kernel: " << op_name
                 << " and hardware: " << (hardware.empty() ? "unknown" : hardware);
  }
}
}  // namespace kernel
}  // namespace mindspore
