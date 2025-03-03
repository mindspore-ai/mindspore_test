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

#ifndef MINDSPORE_SWAP_OUT_ACTOR_H
#define MINDSPORE_SWAP_OUT_ACTOR_H

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "runtime/graph_scheduler/actor/abstract_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/device/gsm/swap_strategy.h"

namespace mindspore {
namespace runtime {
class MemorySwapActor : public AbstractActor {
 public:
  MemorySwapActor(const std::string &name, const AID *recorder_aid, size_t stream_id,
                  std::vector<DeviceTensor *> device_tensors_to_swap)
      : AbstractActor(name, KernelTransformType::kMemorySwapActor, recorder_aid),
        stream_id_(stream_id),
        device_tensors_to_swap_(std::move(device_tensors_to_swap)) {}
  MemorySwapActor(const std::string &name, const AID *recorder_aid, size_t stream_id,
                  std::vector<DeviceTensor *> device_tensors_to_swap, const DeviceContext *device_context,
                  std::vector<std::pair<device::SwapActionType, std::vector<size_t>>> actions)
      : AbstractActor(name, KernelTransformType::kMemorySwapActor, recorder_aid),
        stream_id_(stream_id),
        device_tensors_to_swap_(std::move(device_tensors_to_swap)),
        swap_actions_(std::move(actions)) {
    fixed_device_tensor_num_ = device_tensors_to_swap_.size();
    (void)device_contexts_.emplace_back(device_context);
  }
  ~MemorySwapActor() override = default;

 protected:
  void Run(OpContext<DeviceTensor> *context) override;
  void FetchRealParameters(OpContext<DeviceTensor> *context);

 private:
  void AllocDeviceContinuousMem(const std::vector<DeviceTensor *> &device_tensors);
  static void Swap(OpContext<mindspore::runtime::DeviceTensor> *const context, device::StorageType to,
                   const std::vector<DeviceTensor *> &device_tensors);
  void UpdateDeviceTensors(OpContext<DeviceTensor> *context);
  std::vector<DeviceTensor *> GetDeviceTensors(const std::vector<size_t> &indexes);

 protected:
  size_t stream_id_;
  std::vector<DeviceTensor *> device_tensors_to_swap_;
  std::vector<std::pair<device::SwapActionType, std::vector<size_t>>> swap_actions_;
  std::vector<DeviceTensor *> real_parameters_;
  size_t fixed_device_tensor_num_{0};
};
using MemSwapActorPtr = std::shared_ptr<MemorySwapActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_SWAP_OUT_ACTOR_H
