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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ABSTRACT_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ABSTRACT_ACTOR_H_

#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <set>
#include <unordered_set>
#include <map>
#include "actor/op_actor.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"
#include "backend/ge_backend/runtime/device_tensor_copy_store.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::kernel::KernelTensor;
using mindspore::runtime::ProfilerEvent;
using mindspore::runtime::ProfilerModule;
using mindspore::runtime::ProfilerRecorder;

template <typename T>
using OpData = OpRTData<T>;
template <typename T>
using OpDataUniquePtr = OpRTDataUniquePtr<T>;
template <typename T>
using OpContext = OpRTContext<T>;

// The flag of output data.
constexpr size_t kOutputDataFlagInit = 0;
// Indicates that the output data destination is stack actor, and the output data cannot be reused.
constexpr size_t kOutputDataFlagToStack = 1;

// Counter for callback.
class CallbackCounter {
 public:
  CallbackCounter() = default;
  ~CallbackCounter() = default;

  CallbackCounter(const CallbackCounter &) = delete;
  CallbackCounter &operator=(const CallbackCounter &) = delete;

  size_t Counter() { return counter_.load(); }
  size_t Increase() { return ++counter_; }
  size_t Decrease() { return --counter_; }

  bool expired() const { return expired_.load(); }
  void set_expired(bool expired) { expired_ = expired; }

  void Wait() {
    std::unique_lock<std::mutex> locker(lock_);
    MS_LOG(DEBUG) << "Wait for callback execution start.";
    while (!cv_.wait_for(locker, std::chrono::seconds(1), [&]() { return counter_.load() == 0; })) {
      MS_LOG(DEBUG) << "Wait cycle.";
    }
  }

  void Notify() {
    if (counter_.load() == 0) {
      std::unique_lock<std::mutex> locker(lock_);
      cv_.notify_all();
    }
  }

  std::atomic<size_t> reserved_memory_size_{0};

 private:
  std::atomic<size_t> counter_{0};
  std::mutex lock_;
  std::condition_variable cv_;
  // Callback executed within async thread, this help to indicate that actor is expired.
  std::atomic<bool> expired_{false};
};
using CallbackCounterPtr = std::shared_ptr<CallbackCounter>;

// The abstract common attributes of actors. The actor inheritance relationship:  OpRTActor --> AbstractActor -->
// MemoryAwareActor --> DebugAwareActor --> SuperKernelActor/DataSourceActor/LoopCountActor/OutputActor.
class AbstractActor : public OpRTActor<KernelTensor> {
 public:
  explicit AbstractActor(const std::string &name, KernelTransformType type, const AID *recorder_aid);
  ~AbstractActor() override = default;

  bool IsActive(int msg_num) override { return msg_num >= running_dependent_msg_num_ ? true : false; }

  // The actor run when receive the input data.
  void RunOpData(OpData<KernelTensor> *const input_data, OpContext<KernelTensor> *const context) override;
  // The actor run when receive the input control.
  void RunOpControl(AID *const input_control, OpContext<KernelTensor> *const context) override;

  // Get the position of node in the actor.
  virtual size_t FetchNodePosition(const KernelWithIndex &node) const { return 0; }

  // Get the member.
  KernelTransformType type() const { return type_; }
  int64_t actor_id() const { return actor_id_; }
  const std::vector<AnfNodePtr> &output_data_nodes() const { return output_data_nodes_; }
  const std::vector<std::pair<size_t, AnfNodePtr>> &device_tensor_store_keys() const {
    return device_tensor_store_keys_;
  }
  void set_device_tensor_store_keys(const std::vector<std::pair<size_t, AnfNodePtr>> &device_tensor_store_keys) {
    device_tensor_store_keys_ = device_tensor_store_keys;
  }
  const std::vector<std::pair<AID, DataArrow *>> &input_data_arrow_aids() const { return input_data_arrow_aids_; }
  const std::vector<std::pair<AID, ControlArrow *>> &input_control_arrow_aids() const {
    return input_control_arrow_aids_;
  }
  const std::map<KernelWithIndex, std::vector<AnfNodeWeakPtr>> &internal_parameters() const {
    return internal_parameters_;
  }

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;

  // Check whether satisfy the actor running condition.
  virtual bool CheckRunningCondition(const OpContext<KernelTensor> *context) const;
  // The actor run really when satisfy the actor running condition.
  virtual void Run(OpContext<KernelTensor> *const context) {}

  // Erase input data and input controls when finish actor running.
  virtual void EraseInput(const OpContext<KernelTensor> *context);

  // Fetch input data from the device tensor store.
  void FetchInputByTensorStore(std::vector<KernelTensor *> *const input_launch_tensors,
                               std::vector<KernelTensorPtr> *const input_kernel_tensors,
                               std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
                               std::vector<KernelTensorPtr> *const memory_free_tensors,
                               OpContext<KernelTensor> *const context) const;

  // Init the member output_data_ by output data arrows.
  void InitOutputData();
  // Update the output data before send output data.
  virtual void UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                                const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) {}
  // Send output to downstream actors to trigger running.
  virtual void SendOutput(OpContext<KernelTensor> *const context);
  // Send recorder info to recorder actor.
  virtual void SendRecorderInfo(OpContext<KernelTensor> *const context) const {}
  void SendOutputData(OpContext<KernelTensor> *const context, const std::vector<AnfNodePtr> &output_data_nodes,
                      const std::vector<DataArrowPtr> &output_data_arrows,
                      const std::vector<std::pair<OpDataUniquePtr<KernelTensor>, size_t>> &output_data_list);

  bool IsOutputAddressPersisted(const KernelTensorPtr &output_kernel_tensor, const KernelWithIndex &output_node);

  KernelTransformType type_;

  // The id of recorder actor. Send message to it for recording info.
  const AID *recorder_aid_;

  // Auto increment id for actor.
  int64_t actor_id_;

  // The output_data_nodes_ and output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<AnfNodePtr> output_data_nodes_;
  // The second of pair indicates the output data flag. See constant prefixed with kOutputDataFalg for details.
  std::vector<std::pair<OpDataUniquePtr<KernelTensor>, size_t>> output_data_;

  // When there is recursion in the graph, the actor will send data to the same stack actor multiple times. Since
  // messages are sent asynchronously between actors, there will be multiple messages that remain unprocessed in
  // the channel. In order to prevent old data from being overwritten, it is necessary to allocate a new op data,
  // and these op data will be uniformly cleared by the scheduler after the step ends.
  std::vector<OpDataUniquePtr<KernelTensor>> to_stack_data_;

  // The dependent device tensor stores, the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;

  // The device tensor stores which have the auto monad attribute.
  std::set<AnfNodePtr> auto_monad_device_tensor_stores_;

  // Map <output_node_with_index, internal_parameter> is used to update the shape of internal parameter node for
  // inferring the dynamic shape information of the nodes located at the boundary of the graph partition, such as
  // heterogeneous scenario and so on.
  std::map<KernelWithIndex, std::vector<AnfNodeWeakPtr>> internal_parameters_;

  // The dependent input actors.
  std::vector<std::pair<AID, DataArrow *>> input_data_arrow_aids_;
  std::vector<std::pair<AID, ControlArrow *>> input_control_arrow_aids_;
  // The dependent inputs number.
  size_t input_datas_num_;
  size_t input_controls_num_;

  // The dependent messages number of actor running.
  int running_dependent_msg_num_;
};

using AbstractActorPtr = std::shared_ptr<AbstractActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ABSTRACT_ACTOR_H_
