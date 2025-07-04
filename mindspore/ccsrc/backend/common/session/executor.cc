/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/session/executor.h"
#include "backend/common/session/executor_manager.h"
#include <algorithm>
#include <exception>
#include <set>
#include <utility>
#include "backend/common/backend_common_callback.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/scoped_long_running.h"
#include "frontend/ir/tensor_py.h"

using mindspore::tensor::TensorPybind;
namespace mindspore::session {
namespace {
bool TensorInVector(const VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto &item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      if (TensorInVector(&vector_ref)) {
        return true;
      }
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      return true;
    }
  }
  return false;
}

bool IsTaskReady(const std::shared_ptr<RunGraphTask> &task) {
  MS_EXCEPTION_IF_NULL(task);
  auto session = task->session_;
  MS_EXCEPTION_IF_NULL(session);
  auto graph = session->GetGraph(task->graph_id_);
  if (graph != nullptr) {
    return graph->IsPreGraphFinished();
  }
  return true;
}
}  // namespace

void BuildGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->BuildGraphImpl(graph_id_);
}

void RunGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  MS_LOG(INFO) << "Start run graph " << graph_id_;
  auto graph = session_->GetGraph(graph_id_);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Invalid graph id " << graph_id_;
    return;
  }
  graph->ResetGraphRunningStatus();
  if (AnfUtils::UseMemScheduler()) {
    graph->SetOutputNodeToTensor(node_to_tensor_);
  }
  try {
    session_->LoadInputs(graph_id_, input_tensors_);
    session_->RunGraphImpl(graph_id_, input_tensors_, &outputs_);
    std::map<DeviceAddressPtr, DeviceAddressPtr> new_to_old_device_address;
    session_->UpdateOutputTensors(&outputs_, tensor_to_node_, &new_to_old_device_address);
  } catch (const std::exception &e) {
    session_->ReportErrorMessage();
    ExecutorManager::Instance().OnEvent(ExecutorEvent::kException);
    MsException::Instance().SetException();
  }
  MS_LOG(INFO) << "End run graph " << graph_id_;
  graph->OnRunGraphFinished();
  ExecutorManager::Instance().OnEvent(ExecutorEvent::kRunGraphFinished);
}

void CreateCommGroupTask::Run() { result_ = CommManager::GetInstance().CreateGroupSync(group_name_, ranks_); }

void DestroyCommGroupTask::Run() { result_ = CommManager::GetInstance().DestroyGroup(group_name_); }

Executor::~Executor() {
  try {
    WorkerJoin();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Executor call destructor failed: " << e.what();
  } catch (...) {
    MS_LOG(ERROR) << "Executor call destructor failed.";
  }
}

void Executor::WorkerJoin() {
  // Avoid worker thread join itself which will cause deadlock
  if (worker_->joinable() && worker_->get_id() != std::this_thread::get_id()) {
    {
      std::lock_guard<std::mutex> lock(task_mutex_);
      auto task = std::make_shared<ExitTask>();
      ready_tasks_.push(task);
      task_cond_var_.notify_all();
    }
    worker_->join();
  }
}

void Executor::WorkerLoop() {
  while (true) {
    std::shared_ptr<Task> task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this] { return !ready_tasks_.empty(); });
      task = ready_tasks_.front();
      ready_tasks_.pop();
    }
    MS_EXCEPTION_IF_NULL(task);
    enum TaskType task_type = task->type_;
    bool task_sync_flag = task->sync_run_;
    if (task_type == kExit) {
      OnWorkerExit();
      return;
    }
    try {
      if (task->session_ != nullptr) {
        task->session_->SetThreadContext();
      }
      task->Run();
      if (task->session_ != nullptr) {
        task->session_->ReportWarningMessage();
      }
    } catch (const std::exception &e) {
      if (task->session_ != nullptr) {
        task->session_->ReportErrorMessage();
      }
      ExecutorManager::Instance().OnEvent(ExecutorEvent::kException);
      MsException::Instance().SetException();
    }
    {
      std::lock_guard<std::mutex> lock(done_task_mutex_);
      done_tasks_.emplace_back(std::move(task));
    }
    if (task_type != kRunGraph || task_sync_flag) {
      std::lock_guard<std::mutex> lock(task_mutex_);
      sync_run_task_finished_ = true;
      sync_cond_var_.notify_all();
    }
  }
}

std::vector<std::shared_ptr<RunGraphTask>> Executor::GetReadyTasksFromPendingList() {
  std::vector<std::shared_ptr<RunGraphTask>> ready_tasks;
  std::lock_guard<std::mutex> lock(pending_task_mutex_);
  for (auto iter = pending_tasks_.begin(); iter != pending_tasks_.end();) {
    auto task = *iter;
    if (IsTaskReady(task)) {
      (void)ready_tasks.emplace_back(task);
      iter = pending_tasks_.erase(iter);
    } else {
      ++iter;
    }
  }
  return ready_tasks;
}

void Executor::OnEvent(const ExecutorEvent &event) {
  if (event == ExecutorEvent::kRunGraphFinished) {
    OnRunGraphFinished();
  } else if (event == ExecutorEvent::kClear) {
    OnClear();
  } else if (event == ExecutorEvent::kException) {
    OnException();
  }
}

void Executor::OnClear() {
  {
    mindspore::ScopedLongRunning long_running;
    WorkerJoin();
  }
  ClearDoneTasks();
}

void Executor::OnException() {
  std::vector<std::shared_ptr<Task>> done_tasks;
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    while (!ready_tasks_.empty()) {
      (void)done_tasks.emplace_back(ready_tasks_.front());
      ready_tasks_.pop();
    }
  }
  {
    std::lock_guard<std::mutex> lock(pending_task_mutex_);
    (void)std::copy(pending_tasks_.begin(), pending_tasks_.end(), std::back_inserter(done_tasks));
    pending_tasks_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(done_task_mutex_);
    (void)done_tasks_.insert(done_tasks_.cend(), done_tasks.cbegin(), done_tasks.cend());
  }
}

void Executor::OnRunGraphFinished() {
  auto ready_tasks = GetReadyTasksFromPendingList();
  std::lock_guard<std::mutex> lock(task_mutex_);
  for (auto &task : ready_tasks) {
    ready_tasks_.push(task);
  }
  if (!ready_tasks.empty()) {
    task_cond_var_.notify_all();
  }
  reenter_cond_var_.notify_all();
}

void Executor::ClearDoneTasks() {
  std::lock_guard<std::mutex> lock(done_task_mutex_);
  done_tasks_.clear();
}

void Executor::RunTask(const std::shared_ptr<Task> &task, bool sync, bool long_run) {
  if (sync) {
    ClearDoneTasks();
  }
  {
    std::lock_guard<std::mutex> lock(task_mutex_);
    sync_run_task_finished_ = false;
    ready_tasks_.push(task);
  }
  task_cond_var_.notify_all();
  if (sync && !sync_run_task_finished_) {
    std::unique_lock<std::mutex> lock(task_mutex_);
    if (sync && long_run) {
      mindspore::ScopedLongRunning long_running;
      sync_cond_var_.wait(lock, [this] { return sync_run_task_finished_; });
    } else {
      sync_cond_var_.wait(lock, [this] { return sync_run_task_finished_; });
    }
  }
  ClearDoneTasks();
  MsException::Instance().CheckException();
}

void Executor::BuildGraph(const SessionPtr &session, GraphId graphId) {
  auto task = std::make_shared<BuildGraphTask>();
  task->session_ = session;
  task->graph_id_ = graphId;
  RunTask(task, true);
}

void Executor::RunGraph(const SessionPtr &session, const GraphId &graph_id,
                        const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session);
  MS_EXCEPTION_IF_NULL(outputs);
  auto task = std::make_shared<RunGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  session->CreateOutputTensors(graph_id, inputs, outputs, &task->tensor_to_node_, &task->node_to_tensor_);
  task->outputs_ = *outputs;
  task->sync_run_ = true;
  RunTask(task, true, true);
}

void Executor::RunGraphAsync(const SessionPtr &session, const GraphId &graph_id,
                             const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session);
  MS_EXCEPTION_IF_NULL(outputs);
  auto task = std::make_shared<RunGraphTask>();
  task->session_ = session;
  task->graph_id_ = graph_id;
  task->input_tensors_ = inputs;
  auto graph = session->GetGraph(task->graph_id_);
  if (graph != nullptr && !graph->IsPostGraphFinished()) {
    mindspore::ScopedLongRunning long_running;
    std::unique_lock<std::mutex> lock(reenter_mutex_);
    reenter_cond_var_.wait(lock, [&graph] { return graph->IsPostGraphFinished(); });
    MsException::Instance().CheckException();
  }
  session->CreateOutputTensors(graph_id, inputs, outputs, &task->tensor_to_node_, &task->node_to_tensor_);
  // maintain a copy of output vector
  task->outputs_ = *outputs;

  // Run graph synchronously when the graph require gil.
  if (graph != nullptr && graph->is_need_gil()) {
    std::unique_lock<std::mutex> lock(reenter_mutex_);
    reenter_cond_var_.wait(lock, [&graph] { return graph->IsPreGraphFinished(); });
    MsException::Instance().CheckException();
    task->sync_run_ = true;
    RunTask(task, true, true);
    return;
  }

  // sync run graph without output tensor(int dataset graph)
  MS_EXCEPTION_IF_NULL(graph);
  if ((!TensorInVector(outputs) && !graph->HasPostGraph())) {
    task->sync_run_ = true;
    RunTask(task, true, true);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(pending_task_mutex_);
    if (!IsTaskReady(task)) {
      ClearDoneTasks();
      pending_tasks_.push_back(task);
      return;
    }
  }
  RunTask(task, false);
}

bool Executor::CreateCommGroup(const std::string &group_name, const std::vector<uint32_t> &ranks) {
  auto task = std::make_shared<CreateCommGroupTask>();
  task->group_name_ = group_name;
  task->ranks_ = ranks;
  RunTask(task, true);
  return task->result_;
}

bool Executor::DestroyCommGroup(const std::string &group_name) {
  auto task = std::make_shared<DestroyCommGroupTask>();
  task->group_name_ = group_name;
  RunTask(task, true);
  return task->result_;
}

void Executor::OnWorkerExit() {
  if (device_name_ == kAscendDevice) {
    constexpr char kReleaseKernelRuntimeFunc[] = "ReleaseKernelRuntime";
    static const auto release_kernel_runtime =
      backend_common::BackendCommonCallback::GetInstance().GetCallback<void, const std::string &, uint32_t>(
        kReleaseKernelRuntimeFunc);
    if (release_kernel_runtime) {
      return release_kernel_runtime(kAscendDevice, device_id_);
    }
  }
}
}  // namespace mindspore::session
