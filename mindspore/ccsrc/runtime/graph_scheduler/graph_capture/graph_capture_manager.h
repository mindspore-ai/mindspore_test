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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_GRAPH_CAPTURE_GRAPH_CAPTURE_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_GRAPH_CAPTURE_GRAPH_CAPTURE_MANAGER_H_

#include <vector>
#include <memory>
#include <utility>
#include <queue>
#include <map>
#include <tuple>
#include "runtime/device/res_manager/capture_graph.h"
#include "runtime/graph_scheduler/actor/kernel_runner.h"
#include "runtime/graph_scheduler/graph_parameter_store.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"

namespace mindspore {
namespace runtime {
// The GraphCaptureManager class is used to manage graph capture and replay functionality in kbk mode. It dynamically
// captures kernel launch operations during execution, translates them into a captured graph to sink execution.
// This class provides capabilities for graph capture, replay, and automatic graph partitioning.
using parameter_idx = std::pair<size_t, std::pair<KernelWithIndex, size_t>>;
class BACKEND_EXPORT GraphCaptureManager {
 public:
  static GraphCaptureManager &GetInstance() noexcept;

  // Check whether enable graph capture.
  bool GetEnableGraphCapture() const;
  void SetEnableGraphCapture(bool enable_graph_capture);

  // Check a kernel can be captured or not.
  bool CheckKernelSupportCapture(const KernelRunnerPtr &kernel_runner, const DeviceContext *expected_device_context);

  // According to the execution order, find all operator interval and position that support capture.
  bool FindSupportCaptureKernelPositions(const std::vector<KernelRunnerPtr> &kernel_runners,
                                         const DeviceContext *expected_device_context);

  void Initialize(const DeviceContext *device_context);
  void Reset(const DeviceContext *device_context);

  // Capture operators according to the execution order. Operators that are not supported for capture will be dispatched
  // immediately.
  bool LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                   const std::vector<KernelRunnerPtr> &kernel_runners,
                                   SuperKernelActor *super_kernel_actor, bool hp_mode);
  // Replay all captured sub graphs in series according to the execution order, or execute operators that cannot be
  // captured.
  bool LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                       const std::vector<KernelRunnerPtr> &kernel_runners,
                                       SuperKernelActor *super_kernel_actor, bool hp_mode);

  bool HasCapturedGraph() const { return capture_graph_ && capture_graph_->HasCapturedGraph(); }

  // Before capture graph, process the inputs of all operators. For normal inputs, perform memory solidification
  // by constructing fix_addrs. Record the weights and kv_cache, which will be used during the subsequent replay phase
  // to verify whether there are any changes in the addresses.
  void FetchAllInputsBeforeCaptureGraph(OpContext<KernelTensor> *const context, size_t stream_id,
                                        const std::vector<KernelRunnerPtr> &kernel_runners,
                                        std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  // Through D2D copy operations, update all the fixed ddresses recorded during the capture phase to ensure that
  // the addresses of all normal inputs are valid during the replay phase.
  void UpdateFixAddressBeforeReplayGraph(size_t stream_id, std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  // Using the kv_cache and weight results recorded during the capture phase, verify whether the addresses
  // fetched during replay phase have changed.
  bool CheckParameterNotChange(size_t stream_id);

  void HandleFirstUserMemoryFree(const KernelTensorPtr &kernel_tensor, const KernelRunnerPtr &kernel_actor,
                                 std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  bool IsWeightOrKVCache(GraphParameterStore *cur_graph_parameter_store, const AnfNodePtr &node, size_t parameter_idx);

  void Finalize();

 private:
  enum ExecutorType { CAPTURE_GRAPH = 0, KERNEL };

  GraphCaptureManager() = default;
  ~GraphCaptureManager() = default;
  DISABLE_COPY_AND_ASSIGN(GraphCaptureManager);

  CaptureGraphPtr capture_graph_{nullptr};
  std::vector<CaptureGraphPtr> capture_graphs_;

  // Captured sub graph number.
  size_t capture_graph_num_ = 0;

  // Record all operator interval and position that support capture according to the execution order.
  std::vector<std::pair<size_t, size_t>> capture_kernel_range_positions_;
  // Record all captured sub graphs and kernels that don't support capture, according to the execution order.
  std::vector<std::pair<ExecutorType, size_t>> executors_;

  std::vector<std::tuple<parameter_idx, KernelTensorPtr, KernelRunnerPtr>> fixed_addrs_for_update_;
  std::map<KernelWithIndex, KernelTensorPtr> fixed_addrs_for_set_inputs_;
  std::map<KernelWithIndex, std::tuple<KernelTensorPtr, size_t, KernelRunnerPtr>> weight_kv_addrs_;

  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_GRAPH_CAPTURE_GRAPH_CAPTURE_MANAGER_H_
