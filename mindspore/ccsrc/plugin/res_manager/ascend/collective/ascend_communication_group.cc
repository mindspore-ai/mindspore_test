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
#include <algorithm>
#include "include/backend/distributed/collective/collective_manager.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
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
AscendCommunicationGroup::AscendCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                                   uint32_t global_rank, uint32_t local_group_rank,
                                                   uint32_t local_group_size)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      unique_id_({}),
      comm_(nullptr),
      config_({}) {
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

  bool ret = false;
  std::string rank_table_file_path = common::GetEnv("RANK_TABLE_FILE");
  if (!rank_table_file_path.empty() && !distributed::cluster::ClusterContext::instance()->enable_cross_cluster()) {
    ret = InitializeByRankTable(rank_table_file_path, group_size, group_rank);
  } else {
    ret = InitializeByRootInfoConfig(root_info, group_size, group_rank);
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
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL_WATCHDOG)) {
    MS_LOG(INFO) << "Start initializing hccl watchdog on device side for group: " << name_
                 << ", rank: " << global_rank_;
    HcclWatchDogManager::GetInstance().AddHandler(std::make_unique<HcclWatchDogHandler>(global_rank_, name_, comm_));
    auto handle_size = HcclWatchDogManager::GetInstance().HandleSize();
    (void)HcclWatchDogManager::GetInstance().InitHandler(handle_size);
    MS_LOG(INFO) << "hccl watchdog on device side is successfully initialized.";
  }
  // clear uniqueid
  distributed::collective::CollectiveManager::instance()->ClearUniqueID(name_);
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

void AscendCommunicationGroup::InitializeCommConfig() {
  HcclCommConfigInit(&config_);
  auto instance = distributed::collective::CollectHcclInitInfo::GetInstance();
  uint32_t buffsize = instance->GetBuffsize(name_);
  config_.hcclBufferSize = buffsize == 0 ? HCCL_COMM_DEFAULT_BUFFSIZE : buffsize;
  // The hcclDeterministic configured for the communicator is preferentially based on the parameter in the context,
  // or if this parameter is not configured, hcclDeterministic is configured based on the environment variable
  // 'HCCL_DETERMINISTIC'.
  std::string env_hccl_deterministic = common::GetEnv("HCCL_DETERMINISTIC");
  config_.hcclDeterministic = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON"
                                ? 1
                                : (env_hccl_deterministic == "true" ? 1 : 0);
}

bool AscendCommunicationGroup::InitializeByRootInfoConfig(void *root_info, uint32_t group_size, uint32_t group_rank) {
  InitializeCommConfig();
  MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitRootInfo for " << name_ << ", hcclBufferSize is "
                  << config_.hcclBufferSize << " MB."
                  << " hcclDeterministic is " << config_.hcclDeterministic;
  unique_id_ = *(static_cast<HcclRootInfo *>(root_info));

  HcclCommConfig config;
  HcclCommConfigInit(&config);
  auto ret = strcpy_s(config.hcclCommName, sizeof(config.hcclCommName) / sizeof(config.hcclCommName[0]), name_.c_str());
  if (ret != EOK) {
    MS_LOG(ERROR) << "Call strcpy_s failed, group name: " << name_ << ", error no (" << ret << ")";
    return false;
  }
  if (HcclCommInitRootInfoConfig(static_cast<uint32_t>(group_size), &unique_id_, static_cast<uint32_t>(group_rank),
                                 &config, &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
    MS_LOG(ERROR) << "HcclCommInitRootInfo failed. " + error_message;
    return false;
  }
  MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitRootInfo for " << name_;
  return true;
}

bool AscendCommunicationGroup::InitializeByRankTable(std::string rank_table, uint32_t group_size, uint32_t group_rank) {
  InitializeCommConfig();
  if (name_ == kHCCLGlobalGroupName) {
    // Initialize global communicator by 'HcclCommInitClusterInfoConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitClusterInfoConfig for " << name_
                    << ", hcclBufferSize is " << config_.hcclBufferSize << " MB."
                    << " hcclDeterministic is " << config_.hcclDeterministic;
    if (hccl::HcclAdapter::GetInstance().HcclCommInitClusterInfoConfig(static_cast<const char *>(rank_table.c_str()),
                                                                       static_cast<uint32_t>(global_rank_), &config_,
                                                                       &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
      MS_LOG(ERROR) << "HcclCommInitClusterInfoConfig failed. " + error_message;
      return false;
    }
    MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitClusterInfoConfig for " << name_;
  } else {
    // split sub communicator from global communicator by 'HcclCreateSubCommConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCreateSubCommConfig for " << name_
                    << ", hcclBufferSize is " << config_.hcclBufferSize << " MB."
                    << " hcclDeterministic is " << config_.hcclDeterministic;
    // The HCCL global communicator. This is used as a parameter to segment sub communicator if initializing with
    // 'HcclCreateSubCommConfig'.
    auto global_comm = AscendCollectiveCommLib::GetInstance().HcclCommunicator(kHCCLGlobalGroupName);
    MS_EXCEPTION_IF_NULL(global_comm);
    std::hash<std::string> to_hash;
    size_t sub_comm_id = to_hash(name_);
    if (hccl::HcclAdapter::GetInstance().HcclCreateSubCommConfig(
          &global_comm, static_cast<uint32_t>(group_size), group_ranks_.data(), static_cast<uint64_t>(sub_comm_id),
          static_cast<uint32_t>(group_rank), &config_, &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
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

const HcclComm &AscendCommunicationGroup::hccl_communicator() const { return comm_; }

std::string AscendCommunicationGroup::inner_comm_name() const { return inner_comm_name_; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
