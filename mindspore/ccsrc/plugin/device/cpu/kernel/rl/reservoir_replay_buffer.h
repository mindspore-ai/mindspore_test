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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RESERVOIR_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RESERVOIR_REPLAY_BUFFER_H_

#include <vector>
#include <tuple>
#include <memory>
#include <limits>
#include <random>
#include "common/kernel.h"
#include "utils/log_adapter.h"
#include "plugin/device/cpu/kernel/rl/fifo_replay_buffer.h"

namespace mindspore {
namespace kernel {
class ReservoirReplayBuffer {
 public:
  // Construct a fixed-length reservoir replay buffer.
  ReservoirReplayBuffer(uint32_t seed, size_t capacity, const std::vector<size_t> &schema);
  ~ReservoirReplayBuffer() = default;

  // Push an experience transition to the buffer which will be given the highest reservoir.
  bool Push(const std::vector<KernelTensor *> &transition);

  // Sample a batch transitions with indices and bias correction weights.
  bool Sample(const size_t &batch_size, const std::vector<KernelTensor *> &transition);

 private:
  size_t seed{42};
  size_t capacity_{0};
  size_t total_{0};
  std::vector<size_t> schema_;
  std::default_random_engine generator_;
  // std::uniform_real_distribution<float> dist_{0, 1};
  std::unique_ptr<FIFOReplayBuffer> fifo_replay_buffer_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_RESERVOIR_REPLAY_BUFFER_H_
