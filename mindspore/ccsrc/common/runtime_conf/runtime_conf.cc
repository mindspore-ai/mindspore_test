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
#include "include/common/runtime_conf/runtime_conf.h"
#include "utils/ms_utils.h"
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
      mem_init_size_(kDefaultMemInitSize),
      mem_block_increase_size_(kDefaultMemBlockIncreaseSize),
      mem_max_size_(kDefaultMemMaxSize),
      mem_optimize_level_(0) {}

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

}  // namespace runtime
}  // namespace mindspore
