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

#include "cluster/rpc/tcp/event_loop_group.h"

#include <string>
#include <thread>

#ifdef USE_GLOG
#include "utils/log_adapter.h"
#else
#include "common/log_adapter.h"
#endif

namespace mindspore {
namespace distributed {
namespace rpc {
namespace {
constexpr size_t kOffset_1 = 1;
constexpr size_t kOffset_2 = 2;
constexpr size_t kOffset_4 = 4;
constexpr size_t kOffset_8 = 8;
constexpr size_t kOffset_16 = 16;
constexpr size_t kEventLoopGroupDefaultSize = 8;
static const char kEventLoopGroupNamePrefix[] = "ELGroup-";

size_t CalcFloorSize(size_t size) {
  int ret = size;
  ret |= ret >> kOffset_1;
  ret |= ret >> kOffset_2;
  ret |= ret >> kOffset_4;
  ret |= ret >> kOffset_8;
  ret |= ret >> kOffset_16;
  return (ret + kOffset_1) >> kOffset_1;
}

size_t DeterminateGroupSize(size_t group_size) {
  auto max_size = std::thread::hardware_concurrency();
  if (group_size == 0) {
    MS_LOG(INFO) << "Event loop group threads size is 0, change to default size : " << kEventLoopGroupDefaultSize
                 << ".";
    group_size = kEventLoopGroupDefaultSize;
  }

  size_t size = CalcFloorSize(group_size);
  if (size > max_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "The value of event loop threads size : " << size << " (group size : " << group_size
                               << ") is exceeds the number of CPU cores : " << max_size
                               << ", which is cause performance degradation. This configuration is not legal.";
  }

  return size > kOffset_2 ? size : kOffset_2;
}
}  // namespace

EventLoopGroup::EventLoopGroup(size_t group_size) {
  group_size_ = DeterminateGroupSize(group_size);
  MS_LOG(INFO) << "Event loop group constructed, real threads size : " << group_size_
               << ", configured threads size : " << group_size << ".";
}

EventLoopGroup::~EventLoopGroup() {
  MS_LOG(INFO) << "Event loop group deconstructed.";
  Finalize();
}

void EventLoopGroup::Initialize() {
  MS_LOG(INFO) << "Initialize event loop group.";
  for (size_t i = 0; i < group_size_; i++) {
    EventLoop *event_loop = new (std::nothrow) EventLoop();
    event_loops_.emplace_back(event_loop);
    event_loop->Initialize(kEventLoopGroupNamePrefix + std::to_string(i));
  }
}

void EventLoopGroup::Finalize() {
  MS_LOG(INFO) << "Finalize event loop group.";
  if (event_loops_.size() == 0) {
    return;
  }

  for (size_t i = 0; i < group_size_; i++) {
    auto event_loop = event_loops_[i];
    event_loop->Finalize();
    delete event_loop;
  }
  event_loops_.clear();
}

int EventLoopGroup::SetEventHandler(int fd, uint32_t events, EventHandler handler, void *data) {
  MS_VLOG(VL_DISTRIBUTED_FD) << "Set event handler for fd : " << fd << ".";
  size_t index = static_cast<size_t>(fd) & (group_size_ - 1);
  return event_loops_.at(index)->SetEventHandler(fd, events, handler, data);
}

int EventLoopGroup::DeleteEpollEvent(int fd) {
  MS_VLOG(VL_DISTRIBUTED_FD) << "Delete epoll event fd : " << fd << ".";
  size_t index = static_cast<size_t>(fd) & (group_size_ - 1);
  return event_loops_.at(index)->DeleteEpollEvent(fd);
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
