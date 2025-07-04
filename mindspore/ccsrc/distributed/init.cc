/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/init.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
#include <signal.h>
#endif
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <functional>
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/debug/tft_adapter/tft_wait_sem.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace distributed {
using distributed::recovery::RecoveryContext;
using mindspore::debug::tft::TFTWaitSem;

constexpr char kStopSchedulerFunc[] = "StopRuntimeSchedulerOnException";
constexpr char kOpExecutorWorkerJoinFunc[] = "OpExecutorWorkerJoin";
std::map<std::string, std::function<void()>> gDistributedCallbackMap;

bool Initialize() {
  // If this process participates in the cluster building, we need to initialize cluster context.
  PROF_START(distributed_cluster_init);
  if (common::UseDynamicCluster()) {
    if (!InitializeCluster()) {
      MS_LOG(EXCEPTION)
        << "Failed to initialize distributed job cluster because some processes in the cluster are not successfully "
           "spawned. You can run command: 'grep -rn -E 'ERROR|CRITICAL' -C 10' in your log directory to filter out "
           "error info. It may be one of the following reasons:\n1."
        << kWorkerProcessNotEnoughError << "\n2." << kSchedPortOccupiedError << "\n3."
        << kSchedWorkerAddrNotConsistentError;
    }
  }
  PROF_END(distributed_cluster_init);

  PROF_START(distributed_collective_init);
  // Initialize the collective manager regardless of whether the cluster is initialized or not.
  if (!InitializeCollective()) {
    MS_LOG(EXCEPTION)
      << "Failed to initialize collective communication because some processes in the cluster are not successfully "
         "spawned. You can run command: 'grep -rn -E 'ERROR|CRITICAL' -C 10' in your log directory to filter out error "
         "info.";
  }
  PROF_END(distributed_collective_init);

  // If this is a scheduler node, it does not need to execute other codes like graph compiling and running. We should
  // finalize it immediately.
  if (cluster::ClusterContext::instance()->initialized() &&
      cluster::ClusterContext::instance()->node_role() == kEnvRoleOfScheduler) {
    MS_LOG(INFO) << "Scheduler starts to wait for cluster to exit.";
    (void)cluster::ClusterContext::instance()->Finalize(UINT32_MAX);

    // Release PyNative resources.
    runtime::Pipeline::Get().WaitAll();
    gDistributedCallbackMap[kOpExecutorWorkerJoinFunc]();
    MS_LOG(INFO) << "Scheduler ends waiting for cluster to exit.";
    exit(0);
    return true;
  }

  MsException::Instance().CheckException();
  return true;
}

bool Finalize() {
  if (!FinalizeCollective()) {
    MS_LOG(ERROR) << "Failed to finalize collective communication.";
    return false;
  }

  if (!FinalizeCluster()) {
    MS_LOG(ERROR) << "Failed to finalize cluster.";
    return false;
  }

  return true;
}

bool InitializeCluster() {
  if (!cluster::ClusterContext::instance()->Initialize()) {
    MS_LOG(ERROR) << "Failed to initialize cluster.";
    return false;
  }
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  auto node = cluster::ClusterContext::instance()->node();
  MS_EXCEPTION_IF_NULL(node);

  // Set the callback for the cluster node.
  auto callback = std::make_shared<std::function<void(void)>>([]() {
    MS_LOG(WARNING) << "Callback on exception is called.";
    if (TFTWaitSem::IsEnable()) {
      MS_LOG(INFO) << "Start waiting for TFT.";
      TFTWaitSem::GetInstance().Wait();
      MS_LOG(INFO) << "End waiting for TFT.";
    }

    MS_LOG(DEBUG) << "Start finalizing CollectiveManager in abnormal callback.";
    if (!collective::CollectiveManager::instance()->Finalize()) {
      MS_LOG(EXCEPTION) << "Failed to finalize the collective communication lib.";
    }
    MS_LOG(DEBUG) << "End finalizing CollectiveManager in abnormal callback.";

    if (gDistributedCallbackMap[kStopSchedulerFunc]) {
      gDistributedCallbackMap[kStopSchedulerFunc]();
    }

    MS_LOG(WARNING) << "Kill this process with SIGTERM.";
    // Forcibly Kill this process.
    (void)kill(getpid(), SIGTERM);
  });
  node->set_abnormal_callback(callback);

  if (cluster::ClusterContext::instance()->initialized() && !collective::CollectiveManager::instance()->initialized()) {
    // Scheduler don't use collective communication library.
    const auto &cluster_ctx = cluster::ClusterContext::instance();
    MS_EXCEPTION_IF_NULL(cluster_ctx);
    if (cluster_ctx->node_role() != kEnvRoleOfScheduler) {
      // Global rank id and size should be manually set if cluster is initialized by MindSpore communication framework.
      collective::CollectiveManager::instance()->set_global_rank_id(node->rank_id());
      auto global_rank_size = cluster_ctx->node_num(cluster_ctx->node_role());
      collective::CollectiveManager::instance()->set_global_rank_size(global_rank_size);

      if (RecoveryContext::GetInstance()->enable_recovery()) {
        RecoveryContext::GetInstance()->set_global_rank_id(node->rank_id());
        RecoveryContext::GetInstance()->set_global_rank_size(global_rank_size);
      }
    }
  }
#endif
  return true;
}

bool FinalizeCluster() { return cluster::ClusterContext::instance()->Finalize(); }

bool InitializeCollective() {
  if (collective::CollectiveManager::instance()->initialized()) {
    return true;
  }
  if (cluster::ClusterContext::instance()->initialized() &&
      cluster::ClusterContext::instance()->node_role() == kEnvRoleOfScheduler) {
    MS_LOG(INFO) << "Scheduler node does not need to initialize collective communication.";
    return true;
  }
  if (!collective::CollectiveManager::instance()->Initialize()) {
    return false;
  }

  if (RecoveryContext::GetInstance()->enable_recovery()) {
    RecoveryContext::GetInstance()->ObtainGlobalLatestCkptInfo();
  }
  return true;
}

bool FinalizeCollective() { return collective::CollectiveManager::instance()->Finalize(); }

void set_cluster_exit_with_exception() { cluster::ClusterContext::instance()->set_cluster_exit_with_exception(); }

bool cluster_exit_with_exception() { return cluster::ClusterContext::instance()->cluster_exit_with_exception(); }

void RegisterCallback(const std::string &name, const std::function<void()> &func) {
  gDistributedCallbackMap[name] = func;
}
}  // namespace distributed
}  // namespace mindspore
