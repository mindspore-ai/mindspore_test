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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_SILENT_DETECT_ASCEND_SILENT_CHECK_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_SILENT_DETECT_ASCEND_SILENT_CHECK_H_

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>
#include <string>
#include "ir/dtype/number.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "tools/silent_detect/silent_check/silent_check.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace silentcheck {
namespace ascend {

using Tensor = tensor::Tensor;
using TensorPtr = tensor::TensorPtr;
using kernel::pyboost::OpRunner;

using device::DeviceAddressPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using mindspore::device::DeviceContext;
using TensorPtr = tensor::TensorPtr;

struct DynamicCheckState {
  TensorPtr sfda;  // for SilentCheckV2
  TensorPtr step;  // for SilentCheckV2 and SilentCheckV3
  TensorPtr avg;   // for SilentCheckV3

  bool is_first_call = true;
};
using DynamicCheckStatePtr = std::shared_ptr<DynamicCheckState>;

class CheckObject {
 public:
  CheckObject();
  ~CheckObject() = default;

  void DoSilentCheck(const TensorPtr &input_grad, const DynamicCheckStatePtr &state);
  void DoSilentCheckV2(const TensorPtr &input_grad, const DynamicCheckStatePtr &state);
  void DoSilentCheckV3(const TensorPtr &input_grad, const DynamicCheckStatePtr &state);

  void LaunchNorm(const TensorPtr &input_grad, bool is_l2_norm = true);
  void LaunchSilentCheckV2(const TensorPtr &input_grad, const DynamicCheckStatePtr &state);

  void LaunchSquare();
  void LaunchInplaceCopy(const DynamicCheckStatePtr &state);
  void LaunchSilentCheckV3(const TensorPtr &input_grad, const DynamicCheckStatePtr &state);

 private:
  // operators for both aclnnSilentCheck and aclnnSilentCheckV2
  std::shared_ptr<OpRunner> norm_op_ = nullptr;

  // operators for aclnnSilentCheck
  std::shared_ptr<OpRunner> silent_check_op_ = nullptr;

  // operators for aclnnSilentCheckV2
  std::shared_ptr<OpRunner> square_op_ = nullptr;
  std::shared_ptr<OpRunner> silent_check_v3_op_ = nullptr;
  // operators only used for aclnnSilentCheckV2 first time call
  std::shared_ptr<OpRunner> inplace_copy_op_ = nullptr;
};
using CheckObjPtr = std::shared_ptr<CheckObject>;

class SilentCheckerRegister;

class DynamicSilentChecker : public SilentCheckerBase {
  friend class SilentCheckerRegister;

 public:
  DynamicSilentChecker() = default;

  ~DynamicSilentChecker() override = default;

  void ClearCheckObjects() override { check_objects_.clear(); }

  void Clear() override {
    check_objects_.clear();
    states_.clear();
  }

  bool IsNpuAsdEnable() override;

  bool IsBackProp() { return is_back_prop_; }

  void SetBackProp(bool is_back_prop) override { is_back_prop_ = is_back_prop; }

  void DoSilentCheck(const std::string &op_name, const std::string &comm_group, const TensorPtr &input_grad) override;

  DynamicCheckStatePtr CreateDynamicCheckState(const TensorPtr &input_grad);

 private:
  bool is_back_prop_ = false;
  std::unordered_map<std::string, DynamicCheckStatePtr> states_;
  std::vector<CheckObjPtr> check_objects_;
};

// silent checker implementation for static graph

// SilentCheckV2 computing flow
// [dout] --> L2Norm(aclnnNorm) --> aclnnSilentCheck{step, sfda} --> [comm-operator]

// SilentCheckV3 computing flow
// ====================================================================
// first time call                        | non-first time call
// ---------------------------------------+----------------------------
// [dout]                                 | [dout]
//   v                                    |   v
// InfinityNorm(aclnnNorm)                | InfinityNorm(aclnnNorm)
//   v                                    |   v
// Square(aclnnMul)                       | Square(aclnnMul)
//   |    \                               |   |
//   |     |                              |   |
//   |     v                              |   |
//   |   Copy(aclnnInpalceCopy)           |   |
//   v    /                               |   v
// aclnnSilentCheck{step, avg}            | aclnnSilentCheck{step, avg}
//   v                                    |   v
// [comm-operator]                        | [comm-operator]

struct DeviceAddrInfo {
  DeviceAddressPtr dev_addr;
  size_t max_size;
};

struct OpExecState {
  kernel::KernelModPtr kernel;
  KernelTensorPtr workspace;
  DeviceAddrInfo *output;
  std::string op_name;
};

struct CheckState {
  bool is_first_call = true;

  // for saving state of silent check
  KernelTensorPtr step = nullptr;  // for silent check v2 and v3
  KernelTensorPtr sfda = nullptr;  // for silent check v2
  KernelTensorPtr avg = nullptr;   // for silent check v3

  KernelTensorPtr square = nullptr;
  // In SilentCheckV3, input `val` and `max` are same
  KernelTensorPtr val = nullptr;
  KernelTensorPtr result = nullptr;

  // kernel modules for checking
  OpExecState kernel_norm = {nullptr, nullptr, nullptr};
  OpExecState kernel_square = {nullptr, nullptr, nullptr};
  OpExecState kernel_copy = {nullptr, nullptr, nullptr};

  // used by both SilentCheckV2 and SilentCheckV3
  OpExecState kernel_silent_check = {nullptr, nullptr, nullptr};
};
using CheckStatePtr = std::shared_ptr<CheckState>;

class OPS_ASCEND_API SilentChecker {
 public:
  static SilentChecker &GetInstance();
  static bool IsNpuAsdEnable();
  ~SilentChecker();
  void InitOpExecState(OpExecState *op_exec_state, const std::string &op_name,
                       const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                       DeviceAddrInfo *output);

  void LaunchOperator(const OpExecState *op_exec_state, const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &outputs, KernelTensor *output_tensor, void *stream_ptr);

  void RegisterCheck(const kernel::KernelModPtr &kernel_mod, const kernel::KernelTensor *dout);
  void ClearCheckHooks() { check_states_.clear(); }
  void ExecuteCheck(const kernel::KernelMod *kernel_mod, const kernel::KernelTensor *dout, void *stream_ptr);
  void UpdateDeviceContext(const DeviceContext *device_context) { device_context_ = device_context; }
  void SetCommOpInputNotSupport(bool not_support) { comm_op_input_not_support_ = not_support; }
  bool IsCommOpInputNotSupport() { return comm_op_input_not_support_; }

 private:
  explicit SilentChecker(const DeviceContext *device_context);

  void LaunchNormAsync(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr);
  void LaunchSilentCheckV2Async(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr);

  void LaunchSquareAsync(const CheckStatePtr &state, void *stream_ptr);
  void LaunchInplaceCopyAsync(const CheckStatePtr &state, void *stream_ptr);
  void LaunchSilentCheckV3Async(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr);

  KernelTensorPtr GenerateKernelTensor(TypeId dtype_id, const ShapeVector &shape, const ValuePtr &value = nullptr,
                                       bool alloc_dev = false);

  std::unordered_map<const kernel::KernelMod *, CheckStatePtr> check_states_;
  const DeviceContext *device_context_ = nullptr;
  bool comm_op_input_not_support_ = false;

  // constants used by aclnnNorm
  KernelTensorPtr p_scalar_ = nullptr;
  KernelTensorPtr dim_ = nullptr;
  KernelTensorPtr keep_dim_ = nullptr;

  // constants used by aclnnSilentCheck and aclnnSilentCheckV2
  KernelTensorPtr c_thresh_l1_ = nullptr;     // for silent check v2 and v3
  KernelTensorPtr c_thresh_l2_ = nullptr;     // for silent check v2 and v3
  KernelTensorPtr npu_asd_detect_ = nullptr;  // for silent check v2 and v3
  KernelTensorPtr c_min_steps_ = nullptr;     // for silent check v2
  KernelTensorPtr c_coeff_l1_ = nullptr;      // for silent check v2
  KernelTensorPtr c_coeff_l2_ = nullptr;      // for silent check v2
  KernelTensorPtr beta1_ = nullptr;           // for silent check v3

  // fields for computing
  DeviceAddrInfo out_val_ = {nullptr, 0};     // norm output
  DeviceAddrInfo out_square_ = {nullptr, 0};  // square output
  DeviceAddrInfo out_result_ = {nullptr, 0};  // silent check result

  DeviceAddrInfo workspace_ = {nullptr, 0};  // workspace for silent check related operators
};
}  // namespace ascend
}  // namespace silentcheck
}  // namespace mindspore
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_SILENT_DETECT_ASCEND_SILENT_CHECK_H_
