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
}  // namespace

void BuildGraphTask::Run() {
  MS_EXCEPTION_IF_NULL(session_);
  session_->BuildGraphImpl(graph_id_);
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
    std::lock_guard<std::mutex> lock(done_task_mutex_);
    (void)done_tasks_.insert(done_tasks_.cend(), done_tasks.cbegin(), done_tasks.cend());
  }
}

void Executor::OnRunGraphFinished() { reenter_cond_var_.notify_all(); }

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

void Executor::OnWorkerExit() {}
}  // namespace mindspore::session
