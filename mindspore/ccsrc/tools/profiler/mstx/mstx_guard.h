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
#ifndef MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_GUARD_H_
#define MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_GUARD_H_

#include "tools/profiler/mstx/mstx_impl.h"

namespace mindspore {
namespace profiler {

class MstxRangeGuard {
 public:
  explicit MstxRangeGuard(const char *name, const char *domain)
      : id_(0), enabled_(MstxImpl::GetInstance().IsEnable()), domain_(domain) {
    if (enabled_) {
      MSTX_START(id_, name, nullptr, domain_);
    }
  }

  ~MstxRangeGuard() {
    if (enabled_) {
      MSTX_END(id_, domain_);
    }
  }

  MstxRangeGuard(const MstxRangeGuard &) = delete;
  MstxRangeGuard &operator=(const MstxRangeGuard &) = delete;

 private:
  uint64_t id_;
  bool enabled_;
  const char *domain_;
};

}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_GUARD_H_
