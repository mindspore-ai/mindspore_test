/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include "utils/dlopen_macro.h"
#include "acl/error_codes/rt_error_codes.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/common/debug/common.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const size_t FreeSize = 3ULL * 1024 * 1024 * 1024;
}  // namespace
// There are 2 requirements for enabling collective communication in DVM
// 1. Jit level is set to O1
// 2. At least one of the collective communication primitives is included in `enable_cluster_ops` or
// `enable_cluster_ops_only` of graph kernel flags
bool EnableDvmComm() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b") {
    return false;
  }

  const auto &jit_level = MsContext::GetInstance()->GetJitLevel();
  if (jit_level != "O1") {
    return false;
  }
  const auto &gk_flags = graphkernel::GraphKernelFlags::GetInstance();
  const auto &enable_cluster_ops = gk_flags.enable_cluster_ops;
  const auto &enable_cluster_ops_only = gk_flags.enable_cluster_ops_only;
  auto check_func = [](const std::string &op) {
    return op == "AllReduce" || op == "AllGather" || op == "ReduceScatter";
  };
  if (std::any_of(enable_cluster_ops.begin(), enable_cluster_ops.end(), check_func)) {
    return true;
  }
  if (std::any_of(enable_cluster_ops_only.begin(), enable_cluster_ops_only.end(), check_func)) {
    return true;
  }
  return false;
}

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}

void SaveCopyWeight(const std::vector<tensor::TensorPtr> &copy_weights, const std::vector<AnfNodePtr> &storage_part,
                    aclrtStream stream) {
  MS_LOG(INFO) << "Enable SaveCopyWeight";
  int i = 0;
  for (const auto &node : storage_part) {
    if (!node->isa<Parameter>()) {
      i++;
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      device::DeviceAddressPtr src_device_ptr = nullptr;
      auto src_addr = copy_weights[i]->device_address();
      if (src_addr != nullptr) {
        src_device_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(src_addr);
        MS_EXCEPTION_IF_NULL(src_device_ptr);
      }
      if (src_device_ptr == nullptr || src_device_ptr->GetPtr() == nullptr ||
          IsOneOfHWSpecialFormat(src_device_ptr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        i++;
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, src_device_ptr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
    i++;
  }
}

void StorageWeights(std::vector<tensor::TensorPtr> *copy_weights, const std::vector<AnfNodePtr> &weights,
                    bool *first_save) {
  int i = 0;
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      i++;
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        i++;
        continue;
      }
      const auto &device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      MS_EXCEPTION_IF_NULL(device_ctx);
      device_ctx->Initialize();
      tensor::TensorPtr target_tensor = nullptr;
      if (*first_save) {
        target_tensor = std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape());
        target_tensor->set_device_address(nullptr);
      } else {
        target_tensor = (*copy_weights)[i];
      }

      device_ctx->device_res_manager_->DeviceToDeviceCopy(std::make_shared<tensor::Tensor>(*tensor), target_tensor);
      if (*first_save) {
        copy_weights->push_back(target_tensor);
      }
    }
    i++;
  }
  *first_save = false;
}

size_t GetFreeMemoryInfo() {
  size_t kMegaByte = 1024 * 1024;
  size_t device_hbm_free_size;
  size_t device_hbm_total_size;
  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &device_hbm_free_size, &device_hbm_total_size);
  if (ret != ACL_ERROR_NONE || device_hbm_total_size == 0) {
    MS_LOG(EXCEPTION) << "Internal Error:Get Device MOC memory size failed, ret = " << ret
                      << ", total MOC size:" << device_hbm_total_size;
  }
  MS_LOG(INFO) << "device_hbm_free_size=" << device_hbm_free_size / kMegaByte
               << "MB, device_hbm_total_size=" << device_hbm_total_size / kMegaByte << "MB";
  size_t free_mem_for_save_size = 0;
  if (device_hbm_free_size > FreeSize) {
    free_mem_for_save_size = device_hbm_free_size - FreeSize;
  }
  return free_mem_for_save_size;
}

void SplitWeightsByFreeMemory(const std::vector<AnfNodePtr> &root_weights, std::vector<AnfNodePtr> *prev_part,
                              std::vector<AnfNodePtr> *storage_part, const size_t &free_mem_size_for_save) {
  size_t weights_nums = root_weights.size();
  size_t save_part_size = 0;
  auto index = weights_nums;

  while (index > 0) {
    index--;
    auto param = root_weights[index]->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto tensor_size = static_cast<size_t>(tensor->Size());
      if (save_part_size + tensor_size > free_mem_size_for_save) {
        index++;
        break;
      } else {
        save_part_size += tensor_size;
      }
    }
  }

  if (index > 0) {
    for (size_t i = 0; i < index; i++) {
      prev_part->push_back(root_weights[i]);
    }
  }

  if (index != weights_nums) {
    for (size_t i = 0; i < weights_nums; i++) {
      storage_part->push_back(root_weights[i]);
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
