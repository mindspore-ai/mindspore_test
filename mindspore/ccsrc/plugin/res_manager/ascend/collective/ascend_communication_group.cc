/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "plugin/res_manager/ascend/collective/ascend_communication_group.h"
#include <map>
#include <variant>
#include <unordered_map>
#include <algorithm>
#include "include/backend/distributed/collective/collective_manager.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/res_manager/ascend/hccl_adapter/hccl_adapter.h"
#include "plugin/res_manager/ascend/collective/hccl_watch_dog_thread.h"
#include "plugin/res_manager/ascend/collective/ascend_collective_comm_lib.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "include/backend/distributed/collective/collect_hccl_init_info.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_err_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendCommunicationGroup::AscendCommunicationGroup(
  const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank, uint32_t local_group_rank,
  uint32_t local_group_size, const std::unordered_map<std::string, std::variant<uint32_t, std::string>> &hccl_config)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      unique_id_({}),
      comm_(nullptr),
      hccl_config_(hccl_config) {
  auto ret = memset_s(inner_comm_name_, INNER_COMM_NAME_MAX_LENGTH, 0x00, INNER_COMM_NAME_MAX_LENGTH);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset_s errorno: " << ret;
  }
}

bool AscendCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return true;
  }
  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    // If using hccl CM envs to launch distributed job, no need to call HcclCommInitRootInfo. The group will be
    // initialized in rank table way.
    initialized_ = true;
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  uint32_t group_rank;
  auto group_size = size_;
  if (!common::GetEnv(kSimulationLevel).empty()) {
    group_size = 1;
    group_rank = 0;
  } else {
    group_rank = GetGroupRank(global_rank_);
  }
  // Generate HcclCommConfig.
  HcclCommConfig config = CreateHcclCommConfig();
  // Call HCCL initialize comm API.
  bool ret = false;
  std::string rank_table_file_path = common::GetEnv("RANK_TABLE_FILE");
  if (!rank_table_file_path.empty() && common::GetEnv(kSimulationLevel).empty() &&
      !distributed::cluster::ClusterContext::instance()->enable_cross_cluster()) {
    ret = InitByRankTable(rank_table_file_path, group_size, group_rank, &config);
  } else {
    ret = InitByRootInfoConfig(root_info, group_size, group_rank, config);
  }
  if (!ret) {
    return false;
  }
  // Get HCCL comm name which is used in graph sink mode for GE.
  if (HcclGetCommName(comm_, inner_comm_name_) != static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
    MS_LOG(ERROR) << "HcclGetCommName failed. " + error_message;
    return false;
  }
  initialized_ = true;

  // Initialize watch dog for global communication group.
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL_WATCHDOG) && common::GetEnv(kSimulationLevel).empty()) {
    MS_LOG(INFO) << "Start initializing hccl watchdog on device side for group: " << name_
                 << ", rank: " << global_rank_;
    HcclWatchDogManager::GetInstance().AddHandler(std::make_unique<HcclWatchDogHandler>(global_rank_, name_, comm_));
    auto handle_size = HcclWatchDogManager::GetInstance().HandleSize();
    (void)HcclWatchDogManager::GetInstance().InitHandler(handle_size);
    MS_LOG(INFO) << "hccl watchdog on device side is successfully initialized.";
  }
  distributed::collective::CollectiveManager::instance()->CacheInitedGroups(name_);
  (void)CALL_ASCEND_API(aclrtResetDevice, device_id);
  return true;
}

bool AscendCommunicationGroup::Finalize() {
  if (!initialized_) {
    return true;
  }
  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    // If using hccl CM envs to launch distributed job, comm_ is not initialized. So directly return.
    initialized_ = false;
    return true;
  }

  // This function will be called at a lonesome thread that has no rtContext, so HcclCommDestroy will be failed.
  // Delete these codes when these threads can be bind.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  HcclWatchDogManager::GetInstance().DestroyHandlerByName(name_);
  (void)HcclCommDestroy(comm_);
  (void)CALL_ASCEND_API(aclrtResetDevice, device_id);
  initialized_ = false;
  comm_ = nullptr;
  return true;
}

void AscendCommunicationGroup::InitHcclCommConfig(HcclCommConfig *config) {
  HcclCommConfigInit(config);
  auto instance = distributed::collective::CollectHcclInitInfo::GetInstance();
  uint32_t buffsize = instance->GetBuffsize(name_);
  config->hcclBufferSize = buffsize == 0 ? HCCL_COMM_DEFAULT_BUFFSIZE : buffsize;

  // The hcclDeterministic configured for the communicator is preferentially based on the parameter in the context,
  // or if this parameter is not configured, hcclDeterministic is configured based on the environment variable
  // 'HCCL_DETERMINISTIC'.
  std::string env_hccl_deterministic = common::GetEnv("HCCL_DETERMINISTIC");
  config->hcclDeterministic = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON"
                                ? 1
                                : (env_hccl_deterministic == "true" ? 1 : 0);
}

HcclCommConfig AscendCommunicationGroup::CreateHcclCommConfig() {
  HcclCommConfig config;
  InitHcclCommConfig(&config);

  if (hccl_config_.empty()) {
    return config;
  }

  // Parameters passed to HCCL via API's config have higher priority than those set in the environment variableThe.
  if (hccl_config_.find("hccl_buffer_size") != hccl_config_.end()) {
    if (!std::holds_alternative<uint32_t>(hccl_config_["hccl_buffer_size"])) {
      MS_LOG(EXCEPTION)
        << "Failed to set hcclBufferSize. Type of hccl_buffer_size in GroupOptions should be unsigned integer.";
    }
    config.hcclBufferSize = std::get<uint32_t>(hccl_config_["hccl_buffer_size"]);
  }

  return config;
}

bool AscendCommunicationGroup::InitByRootInfoConfig(void *root_info, uint32_t group_size, uint32_t group_rank,
                                                    const HcclCommConfig &config) {
  MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitRootInfoConfig for " << name_
                  << ", hcclBufferSize is " << config.hcclBufferSize << " MB, hcclDeterministic is "
                  << config.hcclDeterministic;
  unique_id_ = *(static_cast<HcclRootInfo *>(root_info));
  // MindSpore 2.5 uses HcclCommInitRootInfo to avoid using hcclconfig feature.
  if (hccl::HcclAdapter::GetInstance().HcclCommInitRootInfoConfig(static_cast<uint32_t>(group_size), &unique_id_,
                                                                  static_cast<uint32_t>(group_rank), &config,
                                                                  &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
    MS_LOG(ERROR) << "HcclCommInitRootInfoConfig failed. " + error_message;
    return false;
  }
  MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitRootInfoConfig for " << name_;
  return true;
}

bool AscendCommunicationGroup::InitByRankTable(std::string rank_table, uint32_t group_size, uint32_t group_rank,
                                               HcclCommConfig *config) {
  if (name_ == kHCCLGlobalGroupName) {
    // Initialize global communicator by 'HcclCommInitClusterInfoConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitClusterInfoConfig for " << name_
                    << ", hcclBufferSize is " << config->hcclBufferSize << " MB. hcclDeterministic is "
                    << config->hcclDeterministic;
    if (hccl::HcclAdapter::GetInstance().HcclCommInitClusterInfoConfig(static_cast<const char *>(rank_table.c_str()),
                                                                       static_cast<uint32_t>(global_rank_), config,
                                                                       &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
      MS_LOG(ERROR) << "HcclCommInitClusterInfoConfig failed. " + error_message;
      return false;
    }
    MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitClusterInfoConfig for " << name_;
  } else {
    // split sub communicator from global communicator by 'HcclCreateSubCommConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCreateSubCommConfig for " << name_
                    << ", hcclBufferSize is " << config->hcclBufferSize << " MB. hcclDeterministic is "
                    << config->hcclDeterministic;
    // The HCCL global communicator. This is used as a parameter to segment sub communicator if initializing with
    // 'HcclCreateSubCommConfig'.
    auto global_comm = AscendCollectiveCommLib::GetInstance().HcclCommunicator(kHCCLGlobalGroupName);
    MS_EXCEPTION_IF_NULL(global_comm);
    std::hash<std::string> to_hash;
    size_t sub_comm_id = to_hash(name_);
    if (hccl::HcclAdapter::GetInstance().HcclCreateSubCommConfig(
          &global_comm, static_cast<uint32_t>(group_size), group_ranks_.data(), static_cast<uint64_t>(sub_comm_id),
          static_cast<uint32_t>(group_rank), config, &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
      MS_LOG(ERROR) << "HcclCreateSubCommConfig failed. " + error_message;
      return false;
    }
    MS_LOG(WARNING) << "End to initialize communicator by HcclCreateSubCommConfig for " << name_;
  }
  return true;
}

void *AscendCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(unique_id_);
  if (!common::GetEnv(kSimulationLevel).empty() && !hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    if (HcclGetRootInfo(&unique_id_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      return nullptr;
    }
    return &unique_id_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  uint32_t group_rank = GetGroupRank(global_rank_);
  if (group_rank == 0) {
    if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      // If using hccl CM envs to launch distributed job, no need to call HcclGetRootInfo.
      return &unique_id_;
    }
    if (HcclGetRootInfo(&unique_id_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      MS_LOG(ERROR) << "Failed to get HCCL unique id: " << CALL_ASCEND_API(aclGetRecentErrMsg);
      return nullptr;
    }
  }
  return &unique_id_;
}

bool AscendCommunicationGroup::SetGlobalCommInfo(uint32_t master_ip, uint32_t master_port, uint32_t total_rank_size,
                                                 uint32_t node_rank, uint32_t local_rank_size) {
  MS_LOG(WARNING) << "Start to SetGlobalCommInfo for " << name_ << ", master_ip:" << master_ip
                  << ", master_port:" << master_port << ", node_rank:" << node_rank
                  << ", total_rank_size:" << total_rank_size << ", local_rank_size" << local_rank_size;
  int32_t ret = hccl::HcclAdapter::GetInstance().HcclSetGlobalCommInfo(master_ip, master_port, total_rank_size,
                                                                       node_rank, local_rank_size);
  if (ret == static_cast<int32_t>(HCCL_E_NOT_SUPPORT)) {
    MS_LOG(INFO) << "HcclSetGlobalCommInfo is not supported.";
  } else if (ret != static_cast<int32_t>(HCCL_SUCCESS)) {
    MS_LOG(ERROR) << "Failed to set HCCL global comm info: " << CALL_ASCEND_API(aclGetRecentErrMsg);
    return false;
  }
  MS_LOG(WARNING) << "End to SetGlobalCommInfo for " << name_;
  return true;
}

const HcclComm &AscendCommunicationGroup::hccl_communicator() const { return comm_; }

std::string AscendCommunicationGroup::inner_comm_name() const { return inner_comm_name_; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
