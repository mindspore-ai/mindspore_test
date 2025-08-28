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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_EV_LOOP_GROUP_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_RPC_TCP_EV_LOOP_GROUP_H_

#include "cluster/rpc/tcp/event_loop.h"

#include <memory>
#include <vector>

namespace mindspore {
namespace distributed {
namespace rpc {
/**
 * Event loop group designed for connection management.
 * Support only read process currently.
 */
class EventLoopGroup {
 public:
  explicit EventLoopGroup(size_t group_size);
  ~EventLoopGroup();

  void Initialize();

  void Finalize();

  int SetEventHandler(int fd, uint32_t events, EventHandler handler, void *data);

  int DeleteEpollEvent(int fd);

 private:
  size_t group_size_;

  std::vector<EventLoop *> event_loops_;
};
using EventLoopGroupPtr = std::shared_ptr<EventLoopGroup>;
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore

#endif
