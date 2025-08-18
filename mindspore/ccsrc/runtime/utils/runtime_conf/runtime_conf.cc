/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include <algorithm>
#include "utils/ms_context.h"

namespace mindspore {
namespace runtime {
constexpr uint32_t kDefaultDispatchThreadsNum = 5;
constexpr uint32_t kDefaultOpThreadsNum = 25;
constexpr float kDefaultMemInitSize = 2.0;
constexpr float kDefaultMemBlockIncreaseSize = 1.0;
constexpr float kDefaultMemMaxSize = 1024.0;

std::shared_ptr<RuntimeConf> RuntimeConf::inst_context_ = nullptr;
RuntimeConf::RuntimeConf()
    : launch_blocking_(false),
      dispatch_threads_num_(kDefaultDispatchThreadsNum),
      op_threads_num_(kDefaultOpThreadsNum),
      group_launch_thread_num_(0),
      kernel_group_num_(0),
      mem_init_size_(kDefaultMemInitSize),
      mem_block_increase_size_(kDefaultMemBlockIncreaseSize),
      mem_max_size_(kDefaultMemMaxSize),
      mem_optimize_level_(0),
      mem_huge_page_reserve_size_(0.0),
      enable_capture_graph_(false) {}

RuntimeConf::~RuntimeConf() = default;

std::shared_ptr<RuntimeConf> RuntimeConf::GetInstance() {
  static std::once_flag inst_context_init_flag_;
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore RuntimeConf";
      inst_context_ = std::make_shared<RuntimeConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

void ComputeThreadNums(size_t *actor_thread_num, size_t *actor_and_kernel_thread_num) {
  MS_EXCEPTION_IF_NULL(actor_thread_num);
  MS_EXCEPTION_IF_NULL(actor_and_kernel_thread_num);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  const size_t cpu_core_num = std::thread::hardware_concurrency();

  auto runtime_conf_instance = RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  auto inter_op_parallel_num = static_cast<size_t>(context_ptr->get_param<uint32_t>(MS_CTX_INTER_OP_PARALLEL_NUM));
  if (runtime_conf_instance->IsDispatchThreadsNumConfigured()) {
    inter_op_parallel_num = runtime_conf_instance->dispatch_threads_num();
  }
  auto runtime_num_threads = static_cast<size_t>(context_ptr->get_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS));
  if (runtime_conf_instance->IsOpThreadsNumConfigured()) {
    runtime_num_threads = runtime_conf_instance->op_threads_num();
  }

  size_t runtime_num_threads_min = std::min(runtime_num_threads, cpu_core_num);
  size_t inter_op_parallel_num_min = std::min(inter_op_parallel_num, cpu_core_num);
  const float kActorUsage = 0.18;
  const size_t kActorThreadMinNum = 1;
  // Compute the actor and kernel thread num.
  // The MemoryManagerActor binds single thread, so if runtime_num_threads is 30, actor num would be 5,
  // kernel num would be 25.
  if (inter_op_parallel_num_min == 0) {
    size_t actor_thread_max_num =
      std::max(static_cast<size_t>(std::floor(runtime_num_threads_min * kActorUsage)), kActorThreadMinNum);
    *actor_thread_num = actor_thread_max_num;
    *actor_and_kernel_thread_num =
      runtime_num_threads_min > *actor_thread_num ? (runtime_num_threads_min) : (*actor_thread_num + 1);
  } else {
    *actor_thread_num = inter_op_parallel_num_min;
    *actor_and_kernel_thread_num = runtime_num_threads_min + *actor_thread_num;
  }

  if (*actor_and_kernel_thread_num > cpu_core_num) {
    MS_LOG(WARNING) << "The total num of thread pool is " << *actor_and_kernel_thread_num
                    << ", but the num of cpu core is " << cpu_core_num
                    << ", please set the threads within reasonable limits.";
  }
}
}  // namespace runtime
}  // namespace mindspore
