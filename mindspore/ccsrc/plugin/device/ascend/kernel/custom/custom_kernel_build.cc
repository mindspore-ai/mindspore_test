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
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_internal.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
// For kTransDataOpName
#include "mindspore/ops/op_def/array_op_name.h"

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
  k_custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (k_custom_kernel_plugin_ptr == nullptr) {
    return nullptr;
  }
  return k_custom_kernel_plugin_ptr->BuildKernel(anf_node);
}

bool IsRegisteredCustomKernel(const AnfNodePtr &anf_node) {
  k_custom_kernel_plugin_ptr = GetCustomKernelPlugin();
  if (k_custom_kernel_plugin_ptr == nullptr) {
    return false;
  }
  return k_custom_kernel_plugin_ptr->IsRegisteredKernel(anf_node);
}

bool IsEnableCustomNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return IsRegisteredCustomKernel(node);
}

// Helper function to check if an operation is a format-cast operation
static bool IsFormatCastOperation(const std::string &op_name) {
  return op_name == kTransDataOpName || op_name == "FormatCast" || op_name == "NPUFormatCast";
}

// Helper function to process input format for a single input
static std::string ProcessSingleInputFormat(const CNodePtr &kernel, size_t input_index) {
  // Default: use DEFAULT format, only use upstream format for format-cast operations
  std::string inferred_format = kOpFormat_DEFAULT;

  // If the direct input is a format-cast op (e.g., TransData), use its output format
  auto raw_prev_node = common::AnfAlgo::GetPrevNodeOutput(kernel, input_index, false);
  if (raw_prev_node.first != nullptr && raw_prev_node.first->isa<CNode>()) {
    auto raw_prev_op_name = common::AnfAlgo::GetCNodeName(raw_prev_node.first);
    if (IsFormatCastOperation(raw_prev_op_name)) {
      inferred_format = AnfAlgo::GetOutputFormat(raw_prev_node.first, raw_prev_node.second);
      MS_LOG(INFO) << "Input " << input_index << " uses format-cast op '" << raw_prev_op_name
                   << "', adopting its output format: " << inferred_format;
    }
  }

  return inferred_format;
}

// Helper function to process all input formats
static void ProcessInputFormats(const CNodePtr &kernel, std::vector<std::string> *input_formats) {
  for (size_t i = 0; i < input_formats->size(); ++i) {
    (*input_formats)[i] = ProcessSingleInputFormat(kernel, i);
  }
}

// Helper function to apply output formats from mapping
static void ApplyOutputFormats(const std::vector<std::string> &mapping_output_formats,
                               std::vector<std::string> *output_formats) {
  for (size_t i = 0; i < output_formats->size() && i < mapping_output_formats.size(); ++i) {
    if (!mapping_output_formats[i].empty()) {
      (*output_formats)[i] = mapping_output_formats[i];
      MS_LOG(INFO) << "Set output " << i << " format to: " << mapping_output_formats[i];
    }
  }
}

// Helper function to build input formats string for error messages
static std::string BuildInputFormatsString(const std::vector<std::string> &input_formats) {
  std::string input_formats_str = "[";
  for (size_t i = 0; i < input_formats.size(); ++i) {
    if (i > 0) input_formats_str += ", ";
    input_formats_str += input_formats[i];
  }
  input_formats_str += "]";
  return input_formats_str;
}

// Helper function to log successful format mapping
static void LogSuccessfulFormatMapping(const std::string &op_name, const std::vector<std::string> &input_formats,
                                       const std::string &hardware, const std::vector<std::string> &output_formats) {
  if (!output_formats.empty()) {
    MS_LOG(INFO) << "Found matching format mapping for kernel: " << op_name << " with input formats: " << input_formats
                 << " and hardware: " << (hardware.empty() ? "unknown" : hardware);
  } else {
    MS_LOG(INFO) << "Found matching format mapping for inplace kernel: " << op_name
                 << " with input formats: " << input_formats
                 << " and hardware: " << (hardware.empty() ? "unknown" : hardware)
                 << " (no output format mapping needed)";
  }
}

// Helper function to handle format mapping processing
static void ProcessFormatMapping(const std::string &op_name, const std::vector<std::string> &input_formats,
                                 const std::string &hardware, std::vector<std::string> *output_formats) {
  auto &factory = CustomKernelFactory::Instance();

  if (!factory.HasFormatMapping(op_name)) {
    MS_LOG(INFO) << "No format mapping found for custom kernel: " << op_name
                 << " and hardware: " << (hardware.empty() ? "unknown" : hardware);
    return;
  }

  KernelFormatMapping format_mapping;
  bool found_matching = factory.FindMatchingFormatMapping(op_name, input_formats, hardware, &format_mapping);

  if (found_matching) {
    ApplyOutputFormats(format_mapping.output_formats, output_formats);
    LogSuccessfulFormatMapping(op_name, input_formats, hardware, format_mapping.output_formats);
  } else {
    // No matching format mapping found, report error
    std::string input_formats_str = BuildInputFormatsString(input_formats);
    MS_EXCEPTION(ValueError) << "No matching format mapping found for custom kernel: " << op_name
                             << " with input formats: " << input_formats_str
                             << " and hardware: " << (hardware.empty() ? "unknown" : hardware)
                             << ". Please register appropriate format mapping.";
  }
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

  MS_CUSTOM_KERNEL_HARDWARE_FORMAT_MAPPING_REG(
    type_cast, ascend310p,
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT})),
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_FRAC_NZ, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{kOpFormat_FRAC_NZ, kOpFormat_DEFAULT})));

  MS_CUSTOM_KERNEL_HARDWARE_FORMAT_MAPPING_REG(
    type_cast, ascend910b,
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{kOpFormat_DEFAULT, kOpFormat_DEFAULT})),
    MS_FORMAT_MAPPING((std::vector<std::string>{kOpFormat_FRAC_NZ, kOpFormat_DEFAULT}),
                      (std::vector<std::string>{kOpFormat_FRAC_NZ, kOpFormat_DEFAULT})));
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

  // Process input formats using extracted helper function
  ProcessInputFormats(kernel, input_formats);

  // Process format mapping using extracted helper function
  ProcessFormatMapping(op_name, *input_formats, hardware, output_formats);
}
}  // namespace kernel
}  // namespace mindspore
