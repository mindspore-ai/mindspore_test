/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef _MSC_VER
#include <sched.h>
#include <unistd.h>
#endif
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sstream>
#include "thread/threadpool.h"
#include "thread/core_affinity.h"

namespace mindspore {
std::mutex ThreadPool::create_thread_pool_muntex_;

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    alive_ = false;
  }
  cond_var_->notify_one();

  bool terminate = false;
  int count = 0;
  while (local_task_queue_ && !terminate && count++ < kMaxCount) {
    terminate = local_task_queue_->Empty();
    if (!terminate) {
      auto task_split = local_task_queue_->Dequeue();
      (void)TryRunTask(task_split);
    }
  }

  if (thread_->joinable()) {
    thread_->join();
  }
  pool_ = nullptr;
  local_task_queue_ = nullptr;
}

void Worker::CreateThread() { thread_ = std::make_unique<std::thread>(&Worker::Run, this); }

void Worker::ChildAfterFork() {
  THREAD_INFO("worker %ld recreate thread after fork in child process", worker_id_);
  if (cond_var_ != nullptr) {
    (void)cond_var_.release();
    cond_var_ = std::make_unique<std::condition_variable>();
  }
  if (thread_ != nullptr) {
    (void)thread_.release();
    CreateThread();
  }
}

#if defined(BIND_CORE) && !defined(__ANDROID__) && !defined(__APPLE__) && !defined(_MSC_VER) && !defined(_WIN32)
std::string MaskToStr(const cpu_set_t *mask) {
  std::stringstream ss;
  size_t cpu_num = static_cast<size_t>(sysconf(_SC_NPROCESSORS_ONLN));
  for (size_t i = 0; i < cpu_num; i++) {
    ss << (CPU_ISSET(i, mask) ? "1" : "0");
  }
  return ss.str();
}
#endif

void Worker::SetAffinity() {
#ifdef _WIN32
  SetWindowsSelfAffinity(core_id_);
#elif defined(BIND_CORE)
#ifdef __ANDROID__
  int ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %d to cpu failed. ERROR %d", gettid(), errno);
  }
  return;
#else
#if !defined(__APPLE__) && !defined(_MSC_VER)

  THREAD_INFO("Worker pthread_setaffinity_np, mask %s", MaskToStr(&mask_));
  int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask_);
  if (ret != THREAD_OK) {
    THREAD_ERROR("bind thread %lu to cpu failed. ERROR %d", pthread_self(), errno);
  }
  return;
#endif
#endif
#endif
}

std::vector<int> parse_cpu_list(const std::string &cpu_str) {
  std::vector<int> cpus;
  std::stringstream ss(cpu_str);
  std::string item;

  while (std::getline(ss, item, ',')) {
    cpus.push_back(std::stoi(item));
  }
  return cpus;
}

#if defined(__linux__)
void ThreadPool::ThreadPoolSetAffinity(const std::vector<pthread_t> &threads) {
#if defined(BIND_CORE) && !defined(__ANDROID__)
  auto thread_num = threads.size();
  MS_LOG(INFO) << "Start to bind core for actor thread for [" << thread_num << "] threads.";
  auto env_runtime_reserved = std::getenv("CONFIG_BIND_RUNTIME_LIST");
  if (env_runtime_reserved == nullptr) {
    return;
  }

  MS_LOG(WARNING) << "Start to bind core base on CONFIG_BIND_RUNTIME_LIST.";

  std::vector<int> cpu_list = parse_cpu_list(std::string(env_runtime_reserved));
  if (cpu_list.empty()) {
    MS_LOG(WARNING) << "Cpu list is empty, bind core is not enabled.";
    return;
  }
  int ret;
  auto env_enable_fix = std::getenv("ACTOR_THREAD_FIX_BIND");
  if (env_enable_fix != nullptr && (std::string(env_enable_fix) == "false" || std::string(env_enable_fix) == "False")) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (const auto &cpu_id : cpu_list) {
      CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
    }

    for (size_t i = 0; i < thread_num; i++) {
      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << cpu_list << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << cpu_list << " for thread " << threads[i];
      }
    }
  } else if (env_enable_fix != nullptr && (std::string(env_enable_fix) == "cluster")) {
    int cpus_per_thread = cpu_list.size() / thread_num;
    int offset = 0;

    for (size_t i = 0; i < thread_num; i++) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      std::vector<int> sub_list(cpu_list.begin() + offset, cpu_list.begin() + offset + cpus_per_thread);
      for (const auto &cpu_id : sub_list) {
        CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
      }
      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << sub_list << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << sub_list << " for thread " << threads[i];
      }
      offset += cpus_per_thread;
    }
  } else {
    for (size_t i = 0; i < thread_num; i++) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(cpu_list[i % cpu_list.size()], &cpuset);

      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << cpu_list[i % cpu_list.size()] << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << cpu_list[i % cpu_list.size()] << " for thread " << threads[i];
      }
    }
  }
#else
  MS_LOG(ERROR) << "Call not implemented function.";
  MINDRT_EXIT("Not implemented error");
#endif
}

void ThreadPool::APIThreadPoolSetAffinity(const std::vector<pthread_t> &threads, const std::vector<int> &cpu_list,
                                          const std::string actor_thread_fix_bind) {
#if defined(BIND_CORE) && !defined(__ANDROID__)
  auto thread_num = threads.size();
  MS_LOG(INFO) << "Start to bind core for actor thread for [" << thread_num << "] threads.";
  int ret;
  if (!actor_thread_fix_bind.empty() &&
      (std::string(actor_thread_fix_bind) == "true" || std::string(actor_thread_fix_bind) == "True")) {
    for (size_t i = 0; i < thread_num; i++) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(cpu_list[i % cpu_list.size()], &cpuset);

      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << cpu_list[i % cpu_list.size()] << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << cpu_list[i % cpu_list.size()] << " for thread " << threads[i];
      }
    }
  } else if (!actor_thread_fix_bind.empty() && (std::string(actor_thread_fix_bind) == "cluster")) {
    int cpus_per_thread = cpu_list.size() / thread_num;
    int offset = 0;

    for (size_t i = 0; i < thread_num; i++) {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      std::vector<int> sub_list(cpu_list.begin() + offset, cpu_list.begin() + offset + cpus_per_thread);
      for (const auto &cpu_id : sub_list) {
        CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
      }
      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << sub_list << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << sub_list << " for thread " << threads[i];
      }
      offset += cpus_per_thread;
    }
  } else {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (const auto &cpu_id : cpu_list) {
      CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
    }

    for (size_t i = 0; i < thread_num; i++) {
      ret = pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &cpuset);
      if (ret != 0) {
        MS_LOG(WARNING) << "Fail to bind core to " << cpu_list << " for thread " << threads[i];
      } else {
        MS_LOG(WARNING) << "Success to bind core to " << cpu_list << " for thread " << threads[i];
      }
    }
  }
#else
  MS_LOG(ERROR) << "Call not implemented function.";
  MINDRT_EXIT("Not implemented error");
#endif
}
#endif

void Worker::InitWorkerMask(const std::vector<int> &core_list, const size_t workers_size) {
  core_list_ = core_list;
#ifdef _WIN32
  static uint32_t windows_core_index = 0;
  core_id_ = windows_core_index++;
#elif defined(BIND_CORE)
  if (core_list.empty()) {
    return;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (core_list.size() > 0) {
    CPU_SET(core_list[workers_size % core_list.size()], &mask);
  }
  this->set_mask(mask);
#endif
  return;
}

void Worker::Run() {
  if (!core_list_.empty()) {
    SetAffinity();
  }
#if !defined(__APPLE__) && !defined(_MSC_VER)
  static std::atomic_int index = {0};
  (void)pthread_setname_np(pthread_self(), ("KernelThread_" + std::to_string(index++)).c_str());
#endif
#ifdef PLATFORM_86
  // Some CPU kernels need set the flush zero mode to improve performance.
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
  while (alive_) {
    if (RunLocalKernelTask()) {
      spin_count_ = 0;
    } else {
      RunOtherKernelTask();
      YieldAndDeactive();
    }
    if (spin_count_ > max_spin_count_) {
      WaitUntilActive();
      spin_count_ = 1;
    }
  }
}

bool Worker::TryRunTask(TaskSplit *task_split) {
  if (task_split == nullptr) {
    return false;
  }
  auto task = task_split->task_;
  auto task_id = task_split->task_id_;
  task->status |= task->func(task->content, task_id, lhs_scale_, rhs_scale_);
  (void)++task->finished;
  return true;
}

bool Worker::RunLocalKernelTask() {
  bool res = false;
  Task *task = task_.load(std::memory_order_consume);
  if (task != nullptr) {
    int task_id = task_id_.load(std::memory_order_consume);
    task->status |= task->func(task->content, task_id, lhs_scale_, rhs_scale_);
    task_.store(nullptr, std::memory_order_relaxed);
    (void)++task->finished;
    res |= true;
  }

  while (!local_task_queue_->Empty()) {
    auto task_split = local_task_queue_->Dequeue();
    res |= TryRunTask(task_split);
  }
  return res;
}

void Worker::RunOtherKernelTask() {
  if (pool_ == nullptr || pool_->actor_thread_num() <= kMinActorRunOther) {
    return;
  }
  auto queues_length = pool_->task_queues().size();
  for (size_t i = 0; i < queues_length; ++i) {
    size_t index = (worker_id_ + i + 1) % queues_length;
    while (!pool_->task_queues()[index]->Empty()) {
      auto task_split = pool_->task_queues()[index]->Dequeue();
      if (TryRunTask(task_split)) {
        return;
      }
    }
  }
}

void Worker::YieldAndDeactive() {
  // deactivate this worker only on the first entry
  if (spin_count_ == 0) {
    std::lock_guard<std::mutex> _l(mutex_);
    if (local_task_queue_->Empty()) {
      status_.store(kThreadIdle);
    } else {
      return;
    }
  }
  spin_count_++;
  std::this_thread::yield();
}

void Worker::WaitUntilActive() {
  std::unique_lock<std::mutex> _l(mutex_);
  cond_var_->wait(_l, [&] { return status_ == kThreadBusy || active_num_ > 0 || !alive_; });
  if (active_num_ > 0) {
    active_num_--;
  }
  // When active_num > 0, status = kThreadIdle, a task may enqueue,
  // because of spint_count = 0, the status may switch to kThreadIdle without handle this task,
  // then a new task may enqueue to override the old one, which cause task missed.
  // So, after wait, the status_ should be kThreadBusy.
  status_.store(kThreadBusy);
}

void Worker::set_scale(float lhs_scale, float rhs_scale) {
  lhs_scale_ = lhs_scale;
  rhs_scale_ = rhs_scale;
}

void Worker::Active(std::vector<TaskSplit> *task_list, int task_id_start, int task_id_end) {
  {
    std::lock_guard<std::mutex> _l(mutex_);
    // add the first to task_, and others to queue.
    status_ = kThreadBusy;
    Task *task = task_.load(std::memory_order_consume);
    int to_atomic_task = 0;
    if (task == nullptr) {
      task_id_.store(task_id_start, std::memory_order_relaxed);
      THREAD_TEST_TRUE(task_ == nullptr);
      task_.store((*task_list)[0].task_, std::memory_order_release);
      to_atomic_task = 1;
    }
    for (int i = task_id_start + to_atomic_task; i < task_id_end; ++i) {
      while (!local_task_queue_->Enqueue(&(*task_list)[i])) {
      }
    }
    status_ = kThreadBusy;
  }
  cond_var_->notify_one();
}

void Worker::Active() {
  if (active_num_ > 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> _l(mutex_);
    active_num_++;
    status_ = kThreadBusy;
  }
  cond_var_->notify_one();
}

bool Worker::available() {
  int expected = kThreadIdle;
  return status_.compare_exchange_strong(expected, kThreadHeld);
}

ThreadPool::~ThreadPool() {
  for (auto &worker : workers_) {
    delete worker;
    worker = nullptr;
  }
  workers_.clear();

  if (affinity_ != nullptr) {
    delete affinity_;
    affinity_ = nullptr;
  }

  for (auto &task_queue : task_queues_) {
    task_queue->Clean();
  }
  task_queues_.clear();
  THREAD_INFO("destruct success");
}

int ThreadPool::TaskQueuesInit(size_t thread_num) {
  for (size_t i = 0; i < thread_num; ++i) {
    (void)task_queues_.emplace_back(std::make_unique<HQueue<TaskSplit>>());
  }
  for (size_t i = 0; i < thread_num; ++i) {
    if (task_queues_[i]->Init(kMaxHqueueSize) != true) {
      THREAD_ERROR("init task queue failed.");
      return THREAD_ERROR;
    }
  }
  THREAD_INFO("init task queues success.");
  return THREAD_OK;
}

int ThreadPool::ParallelLaunch(const Func &func, Content content, int task_num) {
  // if single thread, run master thread
  if (task_num <= 1) {
    return SyncRunFunc(func, content, 0, task_num);
  }

  // distribute task to the KernelThread and the idle ActorThread,
  // if the task num is greater than the KernelThread num
  THREAD_DEBUG("launch: %d", task_num);
  Task task = {func, content};
  std::vector<TaskSplit> task_list;
  for (int i = 0; i < task_num; ++i) {
    (void)task_list.emplace_back(TaskSplit{&task, i});
  }
  Worker *curr = CurrentWorker();
  DistributeTask(&task_list, &task, task_num, curr);
  // synchronization
  // wait until the finished is equal to task_num
  while (task.finished != task_num) {
    if (curr != nullptr) {
      (void)curr->RunLocalKernelTask();
    }
    std::this_thread::yield();
  }
  // check the return value of task
  if (task.status != THREAD_OK) {
    return THREAD_ERROR;
  }
  return THREAD_OK;
}

void ThreadPool::SyncRunTask(Task *task, int start_num, int task_num) const {
  // run task sequentially
  // if the current thread is not the actor thread
  float per_scale = kMaxScale / (task_num - start_num);
  for (int i = start_num; i < task_num; ++i) {
    float lhs_scale = i * per_scale;
    float rhs_scale = (i + 1) * per_scale;
    rhs_scale = i == task_num - 1 ? kMaxScale : rhs_scale;
    task->status |= task->func(task->content, i, lhs_scale, rhs_scale);
    (void)++task->finished;
  }
}

int ThreadPool::SyncRunFunc(const Func &func, Content content, int start, int end) const {
  for (int i = start; i < end; ++i) {
    int ret = func(content, i, 0, 1);
    if (ret != 0) {
      return ret;
    }
  }
  return THREAD_OK;
}

void ThreadPool::DistributeTask(std::vector<TaskSplit> *task_list, Task *task, int task_num, Worker *curr) const {
  int sum_frequency = 0;
  std::vector<Worker *> assigned;
  assigned.reserve(task_num);
  int num = static_cast<int>(workers_.size()) - 1;
  int offset = 0;
  bool use_curr = (curr != nullptr);
  // if the current thread isn't nullptr, that is the curr is a ActorThread,
  // then assign (task_num - 1) tasks to workers, and run the last one by itself
  int num_assigned = use_curr ? task_num - 1 : task_num;
  int count = 0;

  if (!occupied_actor_thread_) {
    offset = static_cast<int>(actor_thread_num_);
  }

  for (int i = num; i >= offset && count < num_assigned; --i) {
    if (workers_[i]->available()) {
      assigned.push_back(workers_[i]);
      sum_frequency += workers_[i]->frequency();
      (void)++count;
    }
  }

  if (use_curr) {
    assigned.push_back(curr);
    sum_frequency += curr->frequency();
  } else if (assigned.size() != static_cast<size_t>(task_num)) {
    CalculateScales(assigned, sum_frequency);
    ActiveWorkers(assigned, task_list, assigned.size(), curr);
    SyncRunTask(task, assigned.size(), task_num);
    return;
  }

  CalculateScales(assigned, sum_frequency);
  ActiveWorkers(assigned, task_list, task_num, curr);
}

void ThreadPool::CalculateScales(const std::vector<Worker *> &assigned, int sum_frequency) const {
  // divide task according to computing power(core frequency)
  float lhs_scale = 0;
  float rhs_scale = 0;
  if (sum_frequency == 0) {
    return;
  }
  for (const auto &worker : assigned) {
    THREAD_RETURN_IF_NULL(worker);
    rhs_scale += worker->frequency() * 1.0 / sum_frequency;
    rhs_scale = rhs_scale < 1 ? rhs_scale : 1;
    worker->set_scale(lhs_scale, rhs_scale);
    lhs_scale = rhs_scale;
  }
}

void ThreadPool::ActiveWorkers(const std::vector<Worker *> &workers, std::vector<TaskSplit> *task_list, int task_num,
                               const Worker *curr) const {
  // recalculate task num for each worker.
  int worker_num = static_cast<int>(workers.size());
  if (worker_num > 0) {
    int each_worker_task_num = task_num / worker_num;
    int rest_task_num = task_num % worker_num;
    int start = 0;
    int end;
    for (int i = 0; i < worker_num; ++i) {
      Worker *worker = workers[i];
      THREAD_RETURN_IF_NULL(worker);
      if (i < rest_task_num) {
        end = start + each_worker_task_num + 1;
      } else {
        end = start + each_worker_task_num;
      }
      worker->Active(task_list, start, end);
      if (worker == curr) {
        (void)worker->RunLocalKernelTask();
      }
      start = end;
    }
  }
}

void ThreadPool::ActiveWorkers() {
  for (auto &worker : workers_) {
    worker->Active();
  }
}

Worker *ThreadPool::CurrentWorker(size_t *index) const {
  for (*index = 0; *index < workers_.size(); (*index)++) {
    if (workers_[*index]->thread_id() == std::this_thread::get_id()) {
      return workers_[*index];
    }
  }
  return nullptr;
}

Worker *ThreadPool::CurrentWorker() const {
  for (const auto &worker : workers_) {
    if (worker->thread_id() == std::this_thread::get_id()) {
      return worker;
    }
  }
  return nullptr;
}

int ThreadPool::InitAffinityInfo() {
#ifdef BIND_CORE
  affinity_ = new (std::nothrow) CoreAffinity();
  THREAD_ERROR_IF_NULL(affinity_);
  int ret = affinity_->InitHardwareCoreInfo();
  if (ret != THREAD_OK) {
    delete affinity_;
    affinity_ = nullptr;
    return THREAD_ERROR;
  }
#endif

  server_cpu_frequence = CoreAffinity::GetServerFrequency() / 1000.0f;  // 1GHz = 1000MHz

  return THREAD_OK;
}

int ThreadPool::SetCpuAffinity(BindMode bind_mode) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
  if (affinity_ != nullptr) {
    return affinity_->BindThreads(workers_, bind_mode);
  }
  return THREAD_OK;
}

int ThreadPool::SetCpuAffinity(const std::vector<int> &core_list) {
  if (workers_.empty()) {
    return THREAD_ERROR;
  }
  if (affinity_ != nullptr) {
    return affinity_->BindThreads(workers_, core_list);
  }
  return THREAD_OK;
}

int ThreadPool::SetProcessAffinity(BindMode bind_mode) const {
  if (affinity_ != nullptr) {
    return affinity_->BindProcess(bind_mode);
  }
  return THREAD_OK;
}

void ThreadPool::SetKernelThreadMaxSpinCount(int spin_count) {
  size_t num = workers_.size() - 1;
  for (size_t i = num; i >= actor_thread_num_; i--) {
    THREAD_RETURN_IF_NULL(workers_[i]);
    workers_[i]->SetMaxSpinCount(spin_count);
  }
}

void ThreadPool::SetSpinCountMaxValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(max_spin_count_);
  }
  return;
}

void ThreadPool::SetSpinCountMinValue() {
  for (auto worker : workers_) {
    THREAD_RETURN_IF_NULL(worker);
    worker->SetMaxSpinCount(min_spin_count_);
  }
  return;
}

void ThreadPool::SetMaxSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  max_spin_count_ = spin_count;
}

void ThreadPool::SetMinSpinCount(int spin_count) {
  if (spin_count <= 0) {
    return;
  }
  min_spin_count_ = spin_count;
}

ThreadPool *ThreadPool::CreateThreadPool(size_t thread_num, const std::vector<int> &core_list) {
  std::lock_guard<std::mutex> lock(create_thread_pool_muntex_);
  ThreadPool *pool = new (std::nothrow) ThreadPool();
  if (pool == nullptr) {
    return nullptr;
  }
  if (pool->TaskQueuesInit(thread_num) != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  int ret = pool->CreateThreads<Worker>(thread_num, core_list);
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  ret = pool->InitAffinityInfo();
  if (ret != THREAD_OK) {
    delete pool;
    return nullptr;
  }
  return pool;
}

void ThreadPool::SetWorkerIdMap() {
  for (size_t i = 0; i < workers_.size(); ++i) {
    auto thread_id = workers_[i]->thread_id();
    worker_ids_[thread_id] = i;
  }
  return;
}

void ThreadPool::ChildAfterFork() {
  THREAD_INFO("ThreadPool reinitialize workers after fork");
  for (auto &worker : workers_) {
    worker->ChildAfterFork();
  }
}
}  // namespace mindspore
