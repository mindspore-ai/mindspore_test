/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/res_manager/ascend/collective/multi_ascend_communication_group.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
MultiAscendCommunicationGroup::MultiAscendCommunicationGroup(const std::string &name,
                                                             const std::vector<uint32_t> &group_ranks,
                                                             uint32_t global_rank, uint32_t local_group_rank,
                                                             uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      hccl_group_(nullptr),
      lccl_group_(nullptr) {}

bool MultiAscendCommunicationGroup::Initialize(void *root_info) {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl() && lccl_group_ != nullptr) {
    if (!lccl_group_->Initialize(root_info)) {
      MS_LOG(ERROR) << "Failed to initialize LCCL group " << name_;
      return false;
    }
    MS_LOG(INFO) << "Successfully initialize LCCL group " << name_;

    const auto &value = common::GetConfigValue(common::kRuntimeConf, common::kRuntimeCommInitLcclOnly);
    if (value == "1" || value == "true" || value == "True") {
      // Only use LCCL if runtiem dev config 'comm_init_lccl_only' is set to true.
      MS_LOG(INFO) << "This is infer boost, only initialize group for LCCL.";
      return true;
    }
  }
#endif
  if (device::ascend::EnableDvmComm() && dvm_group_ != nullptr) {
    if (!dvm_group_->Initialize(root_info)) {
      MS_LOG(ERROR) << "Failed to initialize DVM communication group";
      return false;
    }
  }
  if (!hccl_group_->Initialize(root_info)) {
    MS_LOG(ERROR) << "Failed to initialize HCCL group " << name_;
    return false;
  }
  MS_LOG(INFO) << "Successfully initialize HCCL group " << name_;
  return true;
}

bool MultiAscendCommunicationGroup::Finalize() {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl() && lccl_group_ != nullptr) {
    if (!lccl_group_->Finalize()) {
      MS_LOG(ERROR) << "Failed to finalize LCCL group" << name_;
      return false;
    }
    MS_LOG(INFO) << "Successfully finalize LCCL group " << name_;
  }
#endif
  if (device::ascend::EnableDvmComm() && dvm_group_ != nullptr) {
    if (!dvm_group_->Finalize()) {
      MS_LOG(ERROR) << "Failed to finalize DVM communication group";
      return false;
    }
  }
  if (!hccl_group_->Finalize()) {
    MS_LOG(ERROR) << "Failed to finalize HCCL group" << name_;
    return false;
  }
  MS_LOG(INFO) << "Successfully finalize HCCL group " << name_;
  return true;
}

void *MultiAscendCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  CommunicationGroupPtr group_to_generate_root_info = nullptr;
  const auto &value = common::GetConfigValue(common::kRuntimeConf, common::kRuntimeCommInitLcclOnly);
  if (value == "1" || value == "true" || value == "True") {
    // Only when this is infer boost and MS_ENABLE_LCCL is set to on, we only use LCCL.
    group_to_generate_root_info = lccl_group_;
  } else {
    group_to_generate_root_info = hccl_group_;
  }
  MS_EXCEPTION_IF_NULL(group_to_generate_root_info);
  void *root_info = group_to_generate_root_info->GenerateRootInfo(root_info_size);
  return root_info;
}

bool MultiAscendCommunicationGroup::SetGlobalCommInfo(uint32_t master_ip, uint32_t master_port,
                                                      uint32_t total_rank_size, uint32_t group_rank,
                                                      uint32_t local_rank_size) {
  const auto &value = common::GetConfigValue(common::kRuntimeConf, common::kRuntimeCommInitLcclOnly);
  if (value == "1" || value == "true" || value == "True") {
    // Only when this is infer boost and MS_ENABLE_LCCL is set to on, we only use LCCL.
    return true;
  }
  if (!hccl_group_->SetGlobalCommInfo(master_ip, master_port, total_rank_size, group_rank, local_rank_size)) {
    MS_LOG(ERROR) << "Failed to SetGlobalCommInfo for HCCL group " << name_;
    return false;
  }
  MS_LOG(INFO) << "Successfully SetGlobalCommInfo for HCCL group " << name_;
  return true;
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
