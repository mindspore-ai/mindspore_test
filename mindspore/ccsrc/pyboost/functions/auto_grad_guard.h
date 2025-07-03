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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_

#include <string>
#include <utility>
#include <memory>
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class OpRunner;
using OpPtr = std::shared_ptr<OpRunner>;
struct PYBOOST_API OpStatus {
  OpStatus();
  OpStatus(bool _disable_mix_precision, bool _is_jit_compiling, size_t _custom_bprop_cell_count,
           std::string device_target)
      : disable_mix_precision(_disable_mix_precision),
        is_jit_compiling(_is_jit_compiling),
        custom_bprop_cell_count(_custom_bprop_cell_count),
        device_target(std::move(device_target)) {}
  bool disable_mix_precision{false};
  bool is_jit_compiling{false};
  size_t custom_bprop_cell_count{0};
  std::string device_target{};
};

class PYBOOST_API OpRunStatus {
 public:
  static OpRunStatus &Get();

  const OpStatus &op_status() { return status_; }
  void set_run_info(OpStatus &&run_info) { status_ = run_info; }

  bool RequireGrad() const { return require_grad_; }
  void SetRequireGrad(bool require_grad) { require_grad_ = require_grad; }
  const std::string &device_target() const { return status_.device_target; }

  void ResetRequireGrad(bool require_grad) { require_grad_ = require_grad; }

  void SetLastOp(const OpPtr &op) { last_op_ = op; }

  OpPtr &&GetLastOp() { return std::move(last_op_); }

 private:
  OpRunStatus() = default;
  ~OpRunStatus() = default;
  DISABLE_COPY_AND_ASSIGN(OpRunStatus);

  OpStatus status_{};
  bool require_grad_{false};
  OpPtr last_op_{nullptr};
};

class PYBOOST_API RequireGradGuard {
 public:
  explicit RequireGradGuard(bool require_grad) {
    origin_require_grad_ = OpRunStatus::Get().RequireGrad();
    OpRunStatus::Get().SetRequireGrad(require_grad);
  }
  ~RequireGradGuard() { OpRunStatus::Get().ResetRequireGrad(origin_require_grad_); }

 private:
  bool origin_require_grad_{false};
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_FUNCTIONS_AUTO_GRAD_GUARD_H_
