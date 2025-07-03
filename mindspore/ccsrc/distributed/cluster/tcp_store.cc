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
#include "include/backend/distributed/cluster/tcp_store.h"
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace distributed {
namespace cluster {
TCPStoreClient::TCPStoreClient() {}
TCPStoreClient::~TCPStoreClient() {}

std::shared_ptr<TCPStoreClient> TCPStoreClient::instance() {
  static std::shared_ptr<TCPStoreClient> instance = nullptr;
  if (instance == nullptr) {
    instance.reset(new (std::nothrow) TCPStoreClient());
    MS_EXCEPTION_IF_NULL(instance);
  }
  return instance;
}

py::bytes TCPStoreClient::GetKey(const std::string &key) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    MS_LOG(EXCEPTION) << "For TCPStoreClient::GetKey, the output shape depends on the actual execution, "
                      << "and it will affect the accuracy of memory in dryrun mode.";
  }
  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    distributed::cluster::ClusterContext::instance()->node_base());
  auto data = cgn->GetMetadata(key);
  return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

void TCPStoreClient::SetKey(const std::string &key, const std::string &value) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return;
  }
  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    distributed::cluster::ClusterContext::instance()->node_base());
  (void)cgn->PutMetadata(key, value, value.size());
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

bool TCPStoreClient::DeleteKey(const std::string &key) {
#if defined(__linux__) && defined(WITH_BACKEND)
  static bool dry_run = common::IsCompileSimulation();
  if (MS_UNLIKELY(dry_run)) {
    return true;
  }
  auto cgn = std::dynamic_pointer_cast<distributed::cluster::topology::ComputeGraphNode>(
    distributed::cluster::ClusterContext::instance()->node_base());
  return cgn->DeleteMetadata(key);
#else
  MS_LOG(EXCEPTION) << "The TCPStore is only supported on linux platform.";
#endif
}

}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
