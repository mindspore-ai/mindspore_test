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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_AVAILABILITY_SILENT_CHECK_DYNAMIC_GRAPH_CHECK_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_AVAILABILITY_SILENT_CHECK_DYNAMIC_GRAPH_CHECK_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/common/pyboost/op_runner.h"
#include "availability/silent_check/silent_check.h"

namespace mindspore {
namespace silentcheck {
namespace ascend {

using BaseTensor = tensor::BaseTensor;
using BaseTensorPtr = tensor::BaseTensorPtr;
using kernel::pyboost::OpRunner;

struct DynamicCheckState {
  BaseTensorPtr sfda;  // for SilentCheckV2
  BaseTensorPtr step;  // for SilentCheckV2 and SilentCheckV3
  BaseTensorPtr avg;   // for SilentCheckV3

  bool is_first_call = true;
};
using DynamicCheckStatePtr = std::shared_ptr<DynamicCheckState>;

class CheckObject {
 public:
  CheckObject();
  ~CheckObject() = default;

  void DoSilentCheck(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state);
  void DoSilentCheckV2(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state);
  void DoSilentCheckV3(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state);

  void LaunchNorm(const BaseTensorPtr &input_grad);
  void LaunchSilentCheckV2(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state);

  void LaunchSquare(const BaseTensorPtr &input_grad);
  void LaunchNeScalar();
  void LaunchMaskedSelect();
  void LaunchMedian(const DynamicCheckStatePtr &state);
  void LaunchMax();
  void LaunchSilentCheckV3(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state);

 private:
  // operators for aclnnSilentCheckV2
  std::shared_ptr<OpRunner> norm_op_ = nullptr;
  std::shared_ptr<OpRunner> silent_check_op_ = nullptr;

  // operators for aclnnSilentCheckV3
  std::shared_ptr<OpRunner> square_op_ = nullptr;
  std::shared_ptr<OpRunner> max_op_ = nullptr;
  std::shared_ptr<OpRunner> silent_check_v3_op_ = nullptr;
  // operators only used for aclnnSilentCheckV3 first time call
  std::shared_ptr<OpRunner> ne_scalar_op_ = nullptr;
  std::shared_ptr<OpRunner> masked_select_op_ = nullptr;
  std::shared_ptr<OpRunner> median_op_ = nullptr;
};
using CheckObjPtr = std::shared_ptr<CheckObject>;

class SilentCheckerRegister;

class DynamicSilentChecker : public SilentCheckerBase {
  friend class SilentCheckerRegister;

 public:
  DynamicSilentChecker() = default;

  ~DynamicSilentChecker() override = default;

  void Clear() override {
    check_objects_.clear();
    states_.clear();
  }

  bool IsNpuAsdEnable() override;

  bool IsBackProp() { return is_back_prop_; }

  void SetBackProp(bool is_back_prop) override {
    is_back_prop_ = is_back_prop;
    if (is_back_prop) {
      check_objects_.clear();
    }
  }

  void DoSilentCheck(const std::string &op_name, const std::string &comm_group,
                     const BaseTensorPtr &input_grad) override;

  DynamicCheckStatePtr CreateDynamicCheckState(const BaseTensorPtr &input_grad);

 private:
  bool is_back_prop_ = false;
  std::unordered_map<std::string, DynamicCheckStatePtr> states_;
  std::vector<CheckObjPtr> check_objects_;
};
}  // namespace ascend
}  // namespace silentcheck
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_AVAILABILITY_SILENT_CHECK_DYNAMIC_GRAPH_CHECK_H_
