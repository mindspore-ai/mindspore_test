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
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore::device::ascend {
bool CustomHitCache(const char *aclnn_api, aclOpExecutor **executor, uint64_t *workspace_size,
                    const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
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
  uint64_t hash_id = CustomAclnnHash(std::string(aclnn_api), inputs, outputs, input_output_types);
  set_hash_key_func(hash_id);
  MS_EXCEPTION_IF_NULL(executor);
  *executor = get_exec_cache_func(hash_id, workspace_size);
  if (*executor == nullptr) {
    return false;
  }
  UninitCacheThreadLocal();
  return true;
}

uint64_t CustomAclnnHash(const std::string &op_type, const std::vector<KernelTensor *> &inputs,
                         const std::vector<KernelTensor *> &outputs,
                         const std::vector<CustomSupportType> &input_output_types) {
  g_hash_offset = 0;
  GatherHash(op_type);
  if ((inputs.size() + outputs.size()) != input_output_types.size()) {
    MS_LOG(EXCEPTION) << "'input_output_types' size" << input_output_types.size()
                      << " is not equal to the sum of the sizes of the input " << inputs.size() << " and output "
                      << outputs.size();
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto type = input_output_types[i];
    switch (type) {
      case CustomSupportType::kTypeTensor: {
        GatherHash(inputs[i]);
        break;
      }
      case CustomSupportType::kTypeBool: {
        GatherHash(inputs[i]->GetValueWithCheck<bool>());
        break;
      }
      case CustomSupportType::kTypeFloat: {
        GatherHash(inputs[i]->GetValueWithCheck<float>());
        break;
      }
      case CustomSupportType::kTypeInt: {
        GatherHash(inputs[i]->GetValueWithCheck<int64_t>());

        break;
      }
      case CustomSupportType::kTypeString: {
        GatherHash(inputs[i]->GetValueWithCheck<std::string>());
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "Custom unsupported input type: " << type;
    }
  }

  for (auto output : outputs) {
    GatherHash(output);
  }
  return calc_hash_id();
}

}  // namespace mindspore::device::ascend
