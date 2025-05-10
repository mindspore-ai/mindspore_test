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

#include <string>
#include "runtime/graph_scheduler/graph_capture/graph_capture_manager.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace runtime {
GraphCaptureManager &GraphCaptureManager::GetInstance() noexcept {
  static GraphCaptureManager instance{};
  return instance;
}

bool GraphCaptureManager::GetEnableGraphCapture() const {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  return runtime_conf_instance->GetEnableKernelLaunchCapture();
}

void GraphCaptureManager::SetEnableGraphCapture(bool enable_graph_capture) {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  runtime_conf_instance->SetEnableKernelLaunchCapture(enable_graph_capture);
}

bool GraphCaptureManager::CheckKernelSupportCapture(const KernelRunnerPtr &kernel_runner,
                                                    const DeviceContext *expected_device_context) {
  MS_EXCEPTION_IF_NULL(kernel_runner);
  const auto &kernel = kernel_runner->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  const auto &kernel_mod = kernel_runner->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);

  auto kernel_type = AnfAlgo::GetKernelType(kernel);
  if (kernel_type == KernelType::ACL_KERNEL) {
    return false;
  }

  if ((kernel_runner->device_contexts())[0]->GetDeviceType() != expected_device_context->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "Capture graph mode can not support cpu kernel: " << kernel->fullname_with_scope();
  }

  if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(EXCEPTION)
      << "Capture graph mode can not support computed depend kernel(whose shape need update after launch.): "
      << kernel->fullname_with_scope();
  }

  auto &llm_manager = LLMManager::GetInstance();
  if (llm_manager.need_force_resize(kernel_mod->kernel_name())) {
    return false;
  }

  return true;
}

bool GraphCaptureManager::FindSupportCaptureKernelPositions(const std::vector<KernelRunnerPtr> &kernel_runners,
                                                            const DeviceContext *expected_device_context) {
  size_t start = 0;
  size_t end = 0;
  bool find_kernel_can_capture = false;
  size_t kernel_num = kernel_runners.size();
  if (kernel_num < 1) {
    return false;
  }
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_runner = kernel_runners[i];
    if (kernel_runner == nullptr) {
      continue;
    }

    if (CheckKernelSupportCapture(kernel_runner, expected_device_context)) {
      if (!find_kernel_can_capture) {
        start = i;
        end = i;
        find_kernel_can_capture = true;
      } else {
        end = i;
      }
    } else {
      if (find_kernel_can_capture) {
        capture_kernel_range_positions_.emplace_back(start, end);
        executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
      }
      executors_.emplace_back(KERNEL, i);
      find_kernel_can_capture = false;
    }
  }

  if (find_kernel_can_capture) {
    capture_kernel_range_positions_.emplace_back(start, end);
    executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
  }

  capture_graph_num_ = capture_kernel_range_positions_.size();
  MS_LOG(INFO) << "Capture graph num: " << capture_graph_num_;
  auto executor_size = executors_.size();
  MS_LOG(DEBUG) << "Dump executor info for capture grpah: ";
  for (size_t i = 0; i < executor_size; i++) {
    std::string executor_mode = (executors_[i].first == CAPTURE_GRAPH ? "capture graph" : "kernel");
    std::ostringstream executor_mode_info;
    if (executors_[i].first == CAPTURE_GRAPH) {
      const auto &range_pair = capture_kernel_range_positions_.at(executors_[i].second);
      executor_mode_info << "executor range:[" << std::to_string(range_pair.first) << ", "
                         << std::to_string(range_pair.second) << "].";
    } else {
      executor_mode_info << "executor order:[" << std::to_string(executors_[i].second) << "]";
    }
    MS_LOG(DEBUG) << "The executor[" << i << "] is " << executor_mode << ", " << executor_mode_info.str();
  }

  return capture_graph_num_ > 0;
}

void GraphCaptureManager::Initialize(const DeviceContext *device_context) {
  if (init_) {
    MS_LOG(EXCEPTION) << "GraphCaptureManager has already initialized.";
  }

  for (size_t i = 0; i < capture_graph_num_; i++) {
    capture_graphs_.push_back(device_context->device_res_manager_->CreateCaptureGraph());
  }
  if (!capture_graphs_.empty()) {
    capture_graph_ = capture_graphs_.front();
    MS_EXCEPTION_IF_NULL(capture_graph_);
  }

  init_ = true;
}

void GraphCaptureManager::ResetCaptureGraphs(const DeviceContext *device_context) {
  if (capture_graph_ && capture_graph_->HasCapturedGraph()) {
    capture_graphs_.clear();

    for (size_t i = 0; i < capture_graph_num_; i++) {
      capture_graphs_.push_back(device_context->device_res_manager_->CreateCaptureGraph());
    }

    if (!capture_graphs_.empty()) {
      capture_graph_ = capture_graphs_.front();
      MS_EXCEPTION_IF_NULL(capture_graph_);
    }
  }
}

bool GraphCaptureManager::LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                                      const std::vector<KernelRunnerPtr> &kernel_runners,
                                                      SuperKernelActor *super_kernel_actor) {
  MS_LOG(INFO) << "Begin launch all kernels with capture graph.";
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      size_t start = capture_kernel_range_positions_[executor.second].first;
      size_t end = capture_kernel_range_positions_[executor.second].second;
      capture_graphs_[executor.second]->CaptureBegin(0);
      MS_LOG(DEBUG) << "Begin captrue graph, executor index: " << i << ", range[" << start << ", " << end << "].";

      for (size_t j = start; j <= end; j++) {
        const auto &kernel_runner = kernel_runners[j];
        if (kernel_runner == nullptr) {
          continue;
        }

        if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
          MS_LOG(ERROR) << "Launch kernel in capture mode failed: " << kernel_runner->kernel()->fullname_with_scope();
          return false;
        }
      }
      capture_graphs_[executor.second]->CaptureEnd(0);
      MS_LOG(DEBUG) << "Begin replay captrue graph, executor index: " << i << ", range[" << start << ", " << end
                    << "].";
      capture_graphs_[executor.second]->ExecuteCaptureGraph(0);
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      MS_LOG(DEBUG) << "Begin launch kernel, executor order index: " << executor.second
                    << ", kernel: " << kernel_runner->kernel()->fullname_with_scope();
      if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
    }
  }
  MS_LOG(INFO) << "End launch all kernels with capture graph.";
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                                          const std::vector<KernelRunnerPtr> &kernel_runners,
                                                          SuperKernelActor *super_kernel_actor) {
  MS_LOG(INFO) << "Begin launch all kernels with replay graph.";
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      capture_graphs_[executor.second]->ExecuteCaptureGraph(0);
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
    }
  }
  MS_LOG(INFO) << "End launch all kernels with replay graph.";
  return true;
}

void GraphCaptureManager::Finalize() {
  capture_graph_ = nullptr;
  if (!capture_graphs_.empty()) {
    capture_graphs_.clear();
  }
}
}  // namespace runtime
}  // namespace mindspore
