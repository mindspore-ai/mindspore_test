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
#include "runtime/device/res_manager/capture_graph.h"
#include "runtime/graph_scheduler/actor/kernel_runner.h"

namespace mindspore {
namespace runtime {
class SuperKernelActor;

// The GraphCaptureManager class is used to manage graph capture and replay functionality in kbk mode. It dynamically
// captures kernel launch operations during execution, translates them into a captured graph to sink execution.
// This class provides capabilities for graph capture, replay, and automatic graph partitioning.
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
  void ResetCaptureGraphs(const DeviceContext *device_context);

  // Capture operators according to the execution order. Operators that are not supported for capture will be dispatched
  // immediately.
  bool LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                   const std::vector<KernelRunnerPtr> &kernel_runners,
                                   SuperKernelActor *super_kernel_actor);
  // Replay all captured sub graphs in series according to the execution order, or execute operators that cannot be
  // captured.
  bool LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                       const std::vector<KernelRunnerPtr> &kernel_runners,
                                       SuperKernelActor *super_kernel_actor);

  bool HasCapturedGraph() const { return capture_graph_ && capture_graph_->HasCapturedGraph(); }

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

  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_GRAPH_CAPTURE_GRAPH_CAPTURE_MANAGER_H_
