/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/data_dump/dump_utils.h"
#endif
#include "ir/dtype/tensor_type.h"

namespace mindspore {
namespace runtime {
using device::DeviceAddressPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchAddr;

// The debug actor is used to debug and dump kernel info, it gets the kernel real time execution info in the device, so
// it is synchronous and blocked.
class DebugActor : public ActorBase {
 public:
  DebugActor() : ActorBase("DebugActor") {}
  ~DebugActor() override = default;

  void ACLDump(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs, bool is_kbyk);

  // The debug of each node.
  void DebugPreLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &op_input_kernel_tensors,
                      const std::vector<KernelTensorPtr> &op_output_kernel_tensors, const DeviceContext *device_context,
                      OpContext<KernelTensor> *const op_context, const AID *from_aid);
  void DebugPostLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &op_input_kernel_tensors,
                       const std::vector<KernelTensorPtr> &op_output_kernel_tensors,
                       const DeviceContext *device_context, OpContext<KernelTensor> *const op_context,
                       const AID *from_aid);
#ifdef ENABLE_DEBUGGER
  void AscendKbkDump(const CNodePtr &cnode, const std::vector<KernelTensor *> &input_kernel_tensors,
                     const std::vector<KernelTensor *> &output_kernel_tensors, const DeviceContext *device_context);
#endif
  void AscendStepStart(const std::vector<KernelGraphPtr> &graphs, std::vector<DeviceContext *> device_contexts);

  void AscendStepEnd();

  // The debug on step begin.
  void DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                        const std::vector<AnfNodePtr> &origin_parameters_order,
                        std::vector<DeviceContext *> device_contexts, OpContext<KernelTensor> *const op_context,
                        const AID *from_aid);

  // The debug on step end.
  void DebugOnStepEnd(OpContext<KernelTensor> *const op_context, const AID *from_aid, int total_running_count_,
                      int sink_size_);
  static inline uint64_t current_step{1};

 private:
  // Print unused Kernel.
  void Finalize() override;

  // class members
  uint32_t exec_order_ = 0;
  int step_count_ = 0;
  bool dump_flag_ = false;
  int is_dataset_sink_ = 0;

  bool profile_started_ = false;
  DeviceContext *device_ctx_ = nullptr;

  // Support multi-thread.
  std::mutex debug_mutex_;
};

}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_DEBUG_ACTOR_H_
