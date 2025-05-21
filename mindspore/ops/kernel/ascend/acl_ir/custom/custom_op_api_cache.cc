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

#include "kernel/ascend/acl_ir/custom/custom_op_api_cache.h"
#include <algorithm>
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore::device::ascend {
bool CustomHitCacheSingle(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size, uint64_t *hash_id,
                          const std::vector<std::vector<KernelTensor *>> &inputs,
                          const std::vector<std::vector<KernelTensor *>> &outputs,
                          const std::vector<CustomSupportType> &input_output_types) {
  static const auto get_exec_cache = device::ascend::GetOpApiFunc("PTAGetExecCache");
  static const auto init_cache_thread_local = device::ascend::GetOpApiFunc("InitPTACacheThreadLocal");
  static const auto set_hash_key = device::ascend::GetOpApiFunc("SetPTAHashKey");
  static const auto can_use_cache = device::ascend::GetOpApiFunc("CanUsePTACache");
  GetExecCache get_exec_cache_func = reinterpret_cast<GetExecCache>(get_exec_cache);
  InitCacheThreadLocal init_cache_thread_local_func = reinterpret_cast<InitCacheThreadLocal>(init_cache_thread_local);
  SetHashKey set_hash_key_func = reinterpret_cast<SetHashKey>(set_hash_key);
  CanUseCache can_use_cache_func = reinterpret_cast<CanUseCache>(can_use_cache);
  bool has_func = get_exec_cache_func && init_cache_thread_local_func && set_hash_key_func;
  bool can_use = can_use_cache_func && can_use_cache_func(aclnn_api);
  if (!has_func || !can_use) {
    return false;
  }
  init_cache_thread_local_func();
  MS_EXCEPTION_IF_NULL(hash_id);
  if (*hash_id == 0) {
    *hash_id = CustomAclnnHash(std::string(aclnn_api), inputs, outputs, input_output_types);
  } else {
    CustomRefreshAddr(std::string(aclnn_api), inputs, outputs, input_output_types);
  }

  set_hash_key_func(*hash_id);
  MS_EXCEPTION_IF_NULL(executor);
  *executor = get_exec_cache_func(*hash_id, workspace_size);
  if (*executor == nullptr) {
    return false;
  }
  UninitCacheThreadLocal();
  return true;
}

uint64_t CustomAclnnHash(const std::string &op_type, const std::vector<std::vector<KernelTensor *>> &inputs,
                         const std::vector<std::vector<KernelTensor *>> &outputs,
                         const std::vector<CustomSupportType> &input_output_types) {
  g_hash_offset = 0;
  GatherHash(op_type);
  if ((inputs.size() + outputs.size()) != input_output_types.size()) {
    MS_LOG(EXCEPTION) << "'input_output_types' size " << input_output_types.size()
                      << " is not equal to the sum of the sizes of the input " << inputs.size() << " and output "
                      << outputs.size();
  }
  std::vector<std::vector<KernelTensor *>> inputs_outputs;
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_outputs));
  std::copy(outputs.begin(), outputs.end(), std::back_inserter(inputs_outputs));
  for (size_t i = 0; i < inputs_outputs.size(); i++) {
    auto dyn_input = inputs_outputs[i];
    KernelTensor *input;
    if (dyn_input.empty()) {
      MS_LOG(EXCEPTION) << "Custom op [" << op_type << "] input-" << i << " is empty!";
    } else {
      input = dyn_input[0];
      MS_EXCEPTION_IF_NULL(input);
    }

    auto type = input_output_types[i];
    MS_LOG(INFO) << "Convert custom op [" << op_type << "] input-" << i
                 << ", input type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Convert custom op [" << op_type << "] input-" << i
                          << ", input type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        GatherHash(input);
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        GatherHash(dyn_input);
        break;
      }
      case CustomSupportType::kTypeBool: {
        GatherHash(device::ascend::ConvertKernelTensor<bool>(input));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        GatherHash(device::ascend::ConvertKernelTensor<float>(input));
        break;
      }
      case CustomSupportType::kTypeDouble: {
        auto value = (input->dtype_id() == kNumberTypeFloat32)
                       ? static_cast<double>(device::ascend::ConvertKernelTensor<float>(input))
                       : device::ascend::ConvertKernelTensor<double>(input);
        GatherHash(value);
        break;
      }
      case CustomSupportType::kTypeInt: {
        GatherHash(device::ascend::ConvertKernelTensor<int64_t>(input));
        break;
      }
      case CustomSupportType::kTypeString: {
        GatherHash(device::ascend::ConvertKernelTensor<std::string>(input));
        break;
      }
      case CustomSupportType::kTypeScalar: {
        GatherHash(device::ascend::ConvertKernelTensor<ScalarPtr>(input));
        break;
      }
      case CustomSupportType::kTypeIntArray: {
        GatherHash(device::ascend::ConvertKernelTensor<std::vector<int64_t>>(input));
        break;
      }
      case CustomSupportType::kTypeBoolArray: {
        GatherHash(device::ascend::ConvertKernelTensor<std::vector<uint8_t>>(input));
        break;
      }
      case CustomSupportType::kTypeFloatArray: {
        GatherHash(device::ascend::ConvertKernelTensor<std::vector<float>>(input));
        break;
      }
      case CustomSupportType::kTypeDType: {
        auto value = input->GetValue();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<Type>()) {
          auto type_id = value->cast<TypePtr>()->type_id();
          GatherHash(type_id);
          break;
        } else {
          MS_LOG(EXCEPTION) << "Kernel tensor' value  is not Type, but is " << value->ToString();
        }
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << static_cast<int64_t>(type);
    }
  }

  return calc_hash_id();
}

void CustomRefreshAddr(const std::string &op_type, const std::vector<std::vector<KernelTensor *>> &inputs,
                       const std::vector<std::vector<KernelTensor *>> &outputs,
                       const std::vector<CustomSupportType> &input_output_types) {
  if ((inputs.size() + outputs.size()) != input_output_types.size()) {
    MS_LOG(EXCEPTION) << "'input_output_types' size " << input_output_types.size()
                      << " is not equal to the sum of the sizes of the input " << inputs.size() << " and output "
                      << outputs.size();
  }
  std::vector<std::vector<KernelTensor *>> inputs_outputs;
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_outputs));
  std::copy(outputs.begin(), outputs.end(), std::back_inserter(inputs_outputs));
  for (size_t i = 0; i < inputs_outputs.size(); i++) {
    auto dyn_input = inputs_outputs[i];
    KernelTensor *input;
    if (dyn_input.empty()) {
      MS_LOG(EXCEPTION) << "Custom op [" << op_type << "] input-" << i << " is empty!";
    } else {
      input = dyn_input[0];
      MS_EXCEPTION_IF_NULL(input);
    }

    auto type = input_output_types[i];
    MS_LOG(DEBUG) << "Convert custom op [" << op_type << "] input-" << i
                  << ", input type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    MS_VLOG(VL_CUSTOM_OP) << "Convert custom op [" << op_type << "] input-" << i
                          << ", input type: " << mindspore::kernel::custom::custom_supported_type_to_string.at(type);
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        RefreshAddr(input);
        break;
      }
      case CustomSupportType::kTypeTensorList: {
        RefreshAddr(dyn_input);
        break;
      }
      case CustomSupportType::kTypeBool: {
        RefreshAddr(device::ascend::ConvertKernelTensor<bool>(input));
        break;
      }
      case CustomSupportType::kTypeFloat: {
        RefreshAddr(device::ascend::ConvertKernelTensor<float>(input));
        break;
      }
      case CustomSupportType::kTypeDouble: {
        auto value = (input->dtype_id() == kNumberTypeFloat32)
                       ? static_cast<double>(device::ascend::ConvertKernelTensor<float>(input))
                       : device::ascend::ConvertKernelTensor<double>(input);
        RefreshAddr(value);
        break;
      }
      case CustomSupportType::kTypeInt: {
        RefreshAddr(device::ascend::ConvertKernelTensor<int64_t>(input));
        break;
      }
      case CustomSupportType::kTypeString: {
        RefreshAddr(device::ascend::ConvertKernelTensor<std::string>(input));
        break;
      }
      case CustomSupportType::kTypeScalar: {
        RefreshAddr(device::ascend::ConvertKernelTensor<ScalarPtr>(input));
        break;
      }
      case CustomSupportType::kTypeIntArray: {
        RefreshAddr(device::ascend::ConvertKernelTensor<std::vector<int64_t>>(input));
        break;
      }
      case CustomSupportType::kTypeBoolArray: {
        RefreshAddr(device::ascend::ConvertKernelTensor<std::vector<uint8_t>>(input));
        break;
      }
      case CustomSupportType::kTypeFloatArray: {
        RefreshAddr(device::ascend::ConvertKernelTensor<std::vector<float>>(input));
        break;
      }
      case CustomSupportType::kTypeDType: {
        auto value = input->GetValue();
        MS_EXCEPTION_IF_NULL(value);
        if (value->isa<Type>()) {
          auto type_id = value->cast<TypePtr>()->type_id();
          RefreshAddr(type_id);
          break;
        } else {
          MS_LOG(EXCEPTION) << "Kernel tensor' value  is not Type, but is " << value->ToString();
        }
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << static_cast<int64_t>(type);
    }
  }
}

}  // namespace mindspore::device::ascend
