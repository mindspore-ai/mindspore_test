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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_

#include <string>
#include <memory>
#include <thread>
#include <vector>
#include <map>
#include <shared_mutex>
#include <limits>
#include "include/backend/distributed/cluster/topology/common.h"
#include "include/backend/distributed/rpc/tcp/tcp_client.h"
#include "include/backend/distributed/cluster/topology/tcp_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The ComputeGraphNode is a separate process representing a sub-graph of the distributed computation graph.
class BACKEND_COMMON_EXPORT ComputeGraphNode : public TcpNodeBase {
 public:
  ComputeGraphNode(const std::string &node_id, const std::string &role)
      : TcpNodeBase(node_id, role), client_ip_(""), authenticated_(false), enable_hb_(false) {
    device_id_ = UINT32_MAX;
    std::string env_device_id = common::GetEnv("DEVICE_ID");
    if (!env_device_id.empty()) {
      if (!common::IsStrNumeric(env_device_id) || std::stoull(env_device_id) > std::numeric_limits<uint32_t>::max()) {
        MS_LOG(WARNING) << "Env 'Device_id' is not set due to invalid input value.";
      } else {
        device_id_ = static_cast<uint32_t>(std::stoul(env_device_id));
      }
    }
  }
  ~ComputeGraphNode() override;

  bool Initialize() override;
  bool Initialized() override;

  bool Finalize(bool force = false) override;

  // Stop the heart beat thread. This method will be invoked when exception happens.
  void StopHeartBeatThread();

  // Exchange metadata(name:value) between all the compute graph nodes.
  // The transaction of the exchange process is guaranteed.
  bool ExchangeMetadata(const std::string &biz, const size_t &rank_size, const std::vector<std::string> &names_prefix,
                        const std::vector<std::string> &values, std::map<std::string, std::string> *results,
                        uint32_t timeout = 90);

  void set_abnormal_callback(std::shared_ptr<std::function<void(void)>> abnormal_callback) override;

  // Return client ip of this cgn which is used for cluster building.
  const std::string &client_ip() const;

 private:
  // Send the register message to the meta server node when this node process startup.
  bool Register();

  // Send the unregister message to the meta server node.
  bool Unregister();

  // Send the heartbeat message to the meta server node.
  bool Heartbeat();

  // Call the `Reconnect` function if the input func execution failed.
  bool ReconnectIfNeeded(const std::function<bool(void)> &func, const std::string &error, size_t retry);
  bool ReconnectWithTimeoutWindow(const std::function<bool(void)> &func, const std::string &error, size_t time_out);

  // Reconnect to the meta server node.
  bool Reconnect();

  // The TCP client used to send heartbeat to meta server.
  std::unique_ptr<rpc::TCPClient> hb_client_;

  // Tcp client ip address of this cgn.
  std::string client_ip_;

  // Incidate whether this node is authenticated by meta server node.
  std::atomic<bool> authenticated_;

  // The heartbeat thread from compute graph node to meta server node.
  std::thread heartbeat_;

  // Indicate whether the heartbeat thread is running.
  std::atomic<bool> enable_hb_;

  // The device id of a single process.
  uint32_t device_id_;

  std::shared_ptr<std::function<void(void)>> abnormal_callback_;

  mutable std::shared_mutex exchange_meta_mutex_;
};
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMPUTE_GRAPH_NODE_H_
