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

#include "kernel/ascend/availability/silent_check/ascend_silent_check.h"
#include <sys/param.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <ios>
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include "availability/silent_check/silent_check.h"
#include "include/common/utils/utils.h"
#include "ir/base_tensor.h"
#include "ir/primal_attr.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "kernel/common/pyboost/auto_generate/max.h"
#include "kernel/common/pyboost/auto_generate/inplace_copy.h"
#include "kernel/common/pyboost/auto_generate/norm.h"
#include "kernel/common/pyboost/auto_generate/silent_check_v2.h"
#include "kernel/common/pyboost/auto_generate/silent_check_v3.h"
#include "kernel/common/pyboost/auto_generate/square.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/kernel.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "op_def/op_name.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "transform/graph_ir/op_adapter_map.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace silentcheck {
namespace ascend {
using mindspore::kernel::pyboost::InplaceCopy;
using mindspore::kernel::pyboost::MemBlock;
using mindspore::kernel::pyboost::Norm;
using mindspore::kernel::pyboost::OpRunner;
using mindspore::kernel::pyboost::PyBoostUtils;
using mindspore::kernel::pyboost::SilentCheckV2;
using mindspore::kernel::pyboost::Square;

using transform::_aclCreateTensor;
using transform::aclOpExecutor;
using transform::aclTensor;
using transform::GetOpApiFunc;

namespace {
constexpr char kNpuAsdEnable[] = "NPU_ASD_ENABLE";
constexpr char kNameSilentCheckV2[] = "aclnnSilentCheck";
constexpr char kNameSilentCheckV3[] = "aclnnSilentCheckV2";
constexpr float kSilentCheckV3Beta = 0.99;
constexpr char kVarNpuAsdUpperThresh[] = "NPU_ASD_UPPER_THRESH";
constexpr char kVarNpuAsdSigmaThresh[] = "NPU_ASD_SIGMA_THRESH";
constexpr char kUpperThreshDefaultVal[] = "1000000,10000";
constexpr char kSigmaThreshDefaultVal[] = "100000,5000";
constexpr float kThreshMinimalVal = 3;
constexpr int64_t kMinSteps = 100;

std::string ltrim(const std::string &str) { return std::regex_replace(str, std::regex("^\\s+"), std::string("")); }

std::string rtrim(const std::string &str) { return std::regex_replace(str, std::regex("\\s+$"), std::string("")); }

std::string trim(const std::string &str) { return ltrim(rtrim(str)); }

std::vector<std::string> split(const std::string &str, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    result.emplace_back(item);
  }

  return result;
}

// parse string in format "value0,value1" satisfying value0 > value1 to two int values and then convert them to float
std::vector<float> parse_thresh(const std::string &value, float min_val) {
  constexpr size_t kNumAsdThreshItems = 2;
  std::vector<float> values;
  auto items = split(value, ',');
  if (items.size() != kNumAsdThreshItems) {
    return values;
  }
  try {
    for (const auto &elem : items) {
      float val = std::stoll(trim(elem));
      if (val < min_val) {
        val = min_val;
      }
      values.push_back(val);
    }
  } catch (std::logic_error const &ex) {
    return {};
  }
  if (values.front() <= values.back()) {
    return {};
  }
  return values;
}

std::vector<float> parse_thresh(const std::string &env_var, const std::string &default_val, float min_val) {
  auto env_value = common::GetEnv(env_var);
  auto values = parse_thresh(env_value, min_val);
  if (!values.empty()) {
    return values;
  }

  if (!env_value.empty()) {
    MS_LOG(WARNING) << "Value of environment var " << env_var << " is invalid, use default value " << default_val
                    << " instead.";
  }

  values = parse_thresh(default_val, min_val);
  if (values.empty()) {
    MS_LOG(EXCEPTION) << "Default value of environment var " << env_var << " is invalid, of which value is "
                      << default_val;
  }
  return values;
}

int GetNpuAsdDetectValue() {
  static auto npu_asd_detect_value = []() -> int {
    auto var_val = common::GetEnv(kNpuAsdEnable);
    if (var_val.size() != 1 || var_val[0] < '0' || var_val[0] > '3') {
      if (!var_val.empty()) {
        MS_LOG(WARNING) << "Valid values of " << kNpuAsdEnable << " are 0, 1, 2 and 3, but got " << var_val << ".";
      }
      return 0;
    }

    int value = var_val[0] - '0';
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Value of environment var `" << kNpuAsdEnable << "` is " << value;
    return value;
  }();
  return npu_asd_detect_value;
}

bool HasApiSilentCheckV3() {
  static bool has_silent_check_v3 = []() {
    auto silent_check_v3 = transform::GetOpApiFunc(kNameSilentCheckV3);
    bool has_v3_api = (silent_check_v3 != nullptr);
    if (!has_v3_api) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Do not has " << kNameSilentCheckV3 << " api, use " << kNameSilentCheckV2
                                      << " instead.";
    }
    return has_v3_api;
  }();
  return has_silent_check_v3;
}

bool IsAsdEnable() {
  static bool is_npu_asd_enable = []() -> bool {
    bool enable_check = true;
    auto ctx = MsContext::GetInstance();
    if (ctx->ascend_soc_version() == kAscendVersion910) {
      enable_check = false;
    } else {
      enable_check = GetNpuAsdDetectValue() > 0;
    }
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Enable silent check is " << std::boolalpha << enable_check
                                    << ", since ascend_soc_version is " << ctx->ascend_soc_version()
                                    << " and environment var " << kNpuAsdEnable << " is " << GetNpuAsdDetectValue();
    return enable_check;
  }();
  return is_npu_asd_enable;
}
}  // namespace

bool DynamicSilentChecker::IsNpuAsdEnable() { return IsAsdEnable(); }

CheckObject::CheckObject() {
  if (HasApiSilentCheckV3()) {
    square_op_ = CREATE_PYBOOST_OP(Square, kAscendDevice);
    max_op_ = CREATE_PYBOOST_OP(Max, kAscendDevice);
    inplace_copy_op_ = CREATE_PYBOOST_OP(InplaceCopy, kAscendDevice);
    silent_check_v3_op_ = CREATE_PYBOOST_OP(SilentCheckV3, kAscendDevice);
  } else {
    norm_op_ = CREATE_PYBOOST_OP(Norm, kAscendDevice);
    silent_check_op_ = CREATE_PYBOOST_OP(SilentCheckV2, kAscendDevice);
  }
}

void CheckObject::DoSilentCheck(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  if (HasApiSilentCheckV3()) {
    DoSilentCheckV3(input_grad, state);
  } else {
    DoSilentCheckV2(input_grad, state);
  }
}

void CheckObject::DoSilentCheckV2(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  LaunchNorm(input_grad);
  LaunchSilentCheckV2(input_grad, state);
}

void CheckObject::DoSilentCheckV3(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  LaunchSquare(input_grad);
  LaunchMax();
  if (state->is_first_call) {
    state->is_first_call = false;
    LaunchInplaceCopy(state);
  }
  LaunchSilentCheckV3(input_grad, state);
}

void CheckObject::LaunchNorm(const BaseTensorPtr &input_grad) {
  auto &op = norm_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto p = std::make_shared<FP32Imm>(2.0);
  auto dim = std::make_shared<ValueTuple>(std::vector<ValuePtr>{});
  auto keepdim = std::make_shared<BoolImm>(false);
  auto dtype = std::make_shared<Int64Imm>(input_grad->Dtype()->type_id());

  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, input_grad, p, dim, keepdim, dtype);
  std::vector<int64_t> dim_vector{};
  ScalarPtr p_scalar = nullptr;
  MAKE_SCALAR(GetValue<float>(p), kNumberTypeFloat32, p_scalar);

  const auto keepdim_imm = GetValue<bool>(keepdim);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_grad);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_grad);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnNorm, device_context, op->stream_id(), input_grad, p_scalar, dim_vector, keepdim_imm,
               outputs[kIndex0]);
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void CheckObject::LaunchSilentCheckV2(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  auto &op = silent_check_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  // create value used by aclnnSilentCheck
  auto val = norm_op_->outputs()[kIndex0];
  auto sfda = state->sfda;
  auto step = state->step;
  auto upper_thresh = parse_thresh(kVarNpuAsdUpperThresh, kUpperThreshDefaultVal, kThreshMinimalVal);
  auto sigma_thresh = parse_thresh(kVarNpuAsdSigmaThresh, kSigmaThreshDefaultVal, kThreshMinimalVal);
  auto c_min_steps_ptr = std::make_shared<Int64Imm>(100);
  auto c_thresh_l1_ptr = std::make_shared<FP32Imm>(upper_thresh.front());
  auto c_coeff_l1_ptr = std::make_shared<FP32Imm>(sigma_thresh.front());
  auto c_thresh_l2_ptr = std::make_shared<FP32Imm>(upper_thresh.back());
  auto c_coeff_l2_ptr = std::make_shared<FP32Imm>(sigma_thresh.back());
  auto npu_asd_detect_ptr = std::make_shared<Int64Imm>(GetNpuAsdDetectValue());

  OpRunner::InferOpOutput(op, val, input_grad, sfda, step, c_min_steps_ptr, c_thresh_l1_ptr, c_coeff_l1_ptr,
                          c_thresh_l2_ptr, c_coeff_l2_ptr, npu_asd_detect_ptr);

  auto c_min_steps = GetValue<int64_t>(c_min_steps_ptr);
  auto c_thresh_l1 = GetValue<pyfloat>(c_thresh_l1_ptr);
  auto c_coeff_l1 = GetValue<pyfloat>(c_coeff_l1_ptr);
  auto c_thresh_l2 = GetValue<pyfloat>(c_thresh_l2_ptr);
  auto c_coeff_l2 = GetValue<pyfloat>(c_coeff_l2_ptr);
  auto npu_asd_detect = GetValue<int64_t>(npu_asd_detect_ptr);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), val, input_grad, sfda, step);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(op->device_context(), val, input_grad, sfda, step);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
  LAUNCH_ACLNN(aclnnSilentCheck, device_context, op->stream_id(), val, input_grad, sfda, step, c_min_steps, c_thresh_l1,
               c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect, op->output(kIndex3));
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void CheckObject::LaunchSilentCheckV3(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  auto &op = silent_check_v3_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  max_op_->outputs()[0]->set_shape(ShapeVector{1});

  auto val = max_op_->outputs()[0];
  auto max = max_op_->outputs()[0];
  auto avg = state->avg;
  auto step = state->step;
  auto upper_thresh = parse_thresh(kVarNpuAsdUpperThresh, kSigmaThreshDefaultVal, kThreshMinimalVal);
  auto c_thresh_l1_ptr = std::make_shared<FP32Imm>(upper_thresh.front());
  auto c_thresh_l2_ptr = std::make_shared<FP32Imm>(upper_thresh.back());
  auto beta_ptr = std::make_shared<FP32Imm>(kSilentCheckV3Beta);
  auto npu_asd_detect_ptr = std::make_shared<Int64Imm>(GetNpuAsdDetectValue());

  OpRunner::InferOpOutput(op, val, max, avg, input_grad, step, c_thresh_l1_ptr, c_thresh_l2_ptr, beta_ptr,
                          npu_asd_detect_ptr);

  auto c_thresh_l1 = GetValue<pyfloat>(c_thresh_l1_ptr);
  auto c_thresh_l2 = GetValue<pyfloat>(c_thresh_l2_ptr);
  auto beta = GetValue<pyfloat>(beta_ptr);
  auto npu_asd_detect = GetValue<int64_t>(npu_asd_detect_ptr);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), val, max, avg, input_grad, step);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(),
                                 std::vector<BaseTensorPtr>{op->output(kIndex3)});

  auto device_context = op->device_context();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(op->device_context(), val, max, avg, input_grad, step);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(op->device_context(), std::vector<BaseTensorPtr>{op->output(kIndex3)});
  LAUNCH_ACLNN(aclnnSilentCheckV2, device_context, op->stream_id(), val, max, avg, input_grad, step,
               input_grad->shape_c(), input_grad->stride(), ShapeVector({input_grad->storage_offset()}), c_thresh_l1,
               c_thresh_l2, beta, npu_asd_detect, op->output(kIndex3));

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void CheckObject::LaunchSquare(const BaseTensorPtr &x_tensor) {
  auto &op = square_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  OpRunner::InferOpOutput(op, x_tensor);
  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, x_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnMul, device_context, op->stream_id(), x_tensor, x_tensor, outputs[0]);

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void CheckObject::LaunchMax() {
  auto &op = max_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto &input_tensor = square_op_->outputs()[kIndex0];

  OpRunner::InferOpOutput(op, input_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnMax, device_context, op->stream_id(), input_tensor, outputs[0]);

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void CheckObject::LaunchInplaceCopy(const DynamicCheckStatePtr &state) {
  auto &op = inplace_copy_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto &src_tensor = max_op_->outputs()[kIndex0];

  OpRunner::InferOpOutput(op, state->avg, src_tensor);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), state->avg, src_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, state->avg, src_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnInplaceCopy, device_context, op->stream_id(), state->avg, src_tensor);

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " end";
}

void DynamicSilentChecker::DoSilentCheck(const std::string &op_name, const std::string &comm_group,
                                         const BaseTensorPtr &input_grad) {
  static bool is_npu_asd_enable = IsAsdEnable();
  if (!is_npu_asd_enable) {
    return;
  }
  if (!is_back_prop_) {
    return;
  }
  // receive op just provide a buffer to store data, so no need to check the data in buffer
  if (op_name == transform::kNameReceive) {
    return;
  }
  auto state_key = op_name + comm_group;
  if (!states_.count(state_key)) {
    states_[state_key] = CreateDynamicCheckState(input_grad);
  }
  auto check_obj = std::make_shared<CheckObject>();
  check_objects_.emplace_back(check_obj);
  check_obj->DoSilentCheck(input_grad, states_[state_key]);
}

DynamicCheckStatePtr DynamicSilentChecker::CreateDynamicCheckState(const BaseTensorPtr &input_grad) {
  auto state = std::make_shared<DynamicCheckState>();
  state->step = std::make_shared<BaseTensor>(kNumberTypeInt64, ShapeVector{1});
  if (HasApiSilentCheckV3()) {
    state->avg = std::make_shared<BaseTensor>(input_grad->data_type(), ShapeVector{1});
  } else {
    state->sfda = std::make_shared<BaseTensor>(kNumberTypeFloat32, ShapeVector{3});
  }
  return state;
}

SILENT_CHECK_REG(kAscendDevice, DynamicSilentChecker);

// silent checker implementation for static graph
SilentChecker::SilentChecker(const DeviceContext *device_context) : device_context_(device_context) {
  if (device_context_ == nullptr) {
    device_context_ = mindspore::device::DeviceContextManager::GetInstance().GetDeviceContext(kAscendDevice).get();
  }
  MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);

  // create constants used by aclnnNorm
  p_scalar_ = std::make_shared<KernelTensor>(nullptr, kTypeNone, kNone);
  dim_ = std::make_shared<KernelTensor>(std::make_shared<abstract::TensorShape>(std::vector<int64_t>{}), kInt64,
                                        MakeValue(std::vector<int64_t>{}));

  keep_dim_ = std::make_shared<KernelTensor>(nullptr, kBool, MakeValue(false));
  zero_ = GenerateKernelTensor(kNumberTypeInt8, ShapeVector{1}, MakeValue<int8_t>(0), true);

  // create constants used by aclnnSilentCheck
  auto upper_thresh = parse_thresh(kVarNpuAsdUpperThresh, kUpperThreshDefaultVal, kThreshMinimalVal);
  auto sigma_thresh = parse_thresh(kVarNpuAsdSigmaThresh, kSigmaThreshDefaultVal, kThreshMinimalVal);
  c_min_steps_ = std::make_shared<KernelTensor>(nullptr, kInt64, MakeValue<int64_t>(kMinSteps));
  c_thresh_l1_ = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue<float>(upper_thresh.front()));
  c_coeff_l1_ = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue<float>(sigma_thresh.front()));
  c_thresh_l2_ = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue<float>(upper_thresh.back()));
  c_coeff_l2_ = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue<float>(sigma_thresh.back()));
  npu_asd_detect_ = std::make_shared<KernelTensor>(nullptr, kInt64, MakeValue<int64_t>(GetNpuAsdDetectValue()));
  beta1_ = std::make_shared<KernelTensor>(nullptr, kFloat32, MakeValue<float>(kSilentCheckV3Beta));
}

SilentChecker &SilentChecker::GetInstance() {
  static std::unique_ptr<SilentChecker> inst_ptr = nullptr;
  if (inst_ptr == nullptr) {
    inst_ptr.reset(new SilentChecker(nullptr));
    MS_EXCEPTION_IF_NULL(inst_ptr);
  }
  return *inst_ptr;
}

SilentChecker::~SilentChecker() {}

void SilentChecker::RegisterCheck(const kernel::KernelModPtr &kernel_mod, const kernel::KernelTensor *dout) {
  auto state = std::make_shared<CheckState>();
  check_states_[kernel_mod.get()] = state;

  state->is_first_call = true;
  state->step = GenerateKernelTensor(kNumberTypeInt64, ShapeVector{1}, MakeValue<int64_t>(0), true);
  if (HasApiSilentCheckV3()) {
    state->avg = GenerateKernelTensor(dout->dtype_id(), ShapeVector{1}, nullptr, true);
    state->val = GenerateKernelTensor(dout->dtype_id(), ShapeVector{1});
    state->square = GenerateKernelTensor(dout->dtype_id(), dout->GetShapeVector());
    out_square_.max_size = std::max(out_square_.max_size, state->square->size());
  } else {
    state->sfda = GenerateKernelTensor(kNumberTypeFloat32, ShapeVector{3}, nullptr, true);
    state->val = GenerateKernelTensor(dout->dtype_id(), ShapeVector{});
  }
  state->result = GenerateKernelTensor(kNumberTypeInt32, ShapeVector{1});
  out_val_.max_size = std::max(out_val_.max_size, state->val->size());
  out_result_.max_size = std::max(out_result_.max_size, state->result->size());

  // create kernel modules
  if (HasApiSilentCheckV3()) {
    // square
    InitOpExecState(&state->kernel_square, ops::kNameSquare, {const_cast<KernelTensor *>(dout)}, {state->square.get()},
                    &out_square_);
    // max
    InitOpExecState(&state->kernel_max, ops::kNameMax, {state->square.get()}, {state->val.get()}, &out_val_);
    // inplace copy
    InitOpExecState(&state->kernel_copy, ops::kNameInplaceCopy, {state->avg.get(), state->val.get()}, {}, &out_val_);
    // silent check v3
    InitOpExecState(&state->kernel_silent_check, ops::kNameSilentCheckV3,
                    {state->val.get(), state->val.get(), state->avg.get(), const_cast<KernelTensor *>(dout),
                     state->step.get(), c_thresh_l1_.get(), c_thresh_l2_.get(), beta1_.get(), npu_asd_detect_.get()},
                    {state->val.get(), const_cast<KernelTensor *>(dout), state->step.get(), state->result.get()},
                    &out_result_);
  } else {
    // norm
    InitOpExecState(&state->kernel_norm, ops::kNameNorm,
                    {const_cast<KernelTensor *>(dout), p_scalar_.get(), dim_.get(), keep_dim_.get()},
                    {state->val.get()}, &out_val_);
    // silent_check_v2
    InitOpExecState(
      &state->kernel_silent_check, ops::kNameSilentCheckV2,
      {state->val.get(), const_cast<KernelTensor *>(dout), state->sfda.get(), state->step.get(), c_min_steps_.get(),
       c_thresh_l1_.get(), c_coeff_l1_.get(), c_thresh_l2_.get(), c_coeff_l2_.get(), npu_asd_detect_.get()},
      {const_cast<KernelTensor *>(dout), state->sfda.get(), state->step.get(), state->result.get()}, &out_result_);
  }
}

void SilentChecker::InitOpExecState(OpExecState *op_exec_state, const std::string &op_name,
                                    const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs, DeviceAddrInfo *output) {
  MS_EXCEPTION_IF_NULL(op_exec_state);
  auto kernel_mod = device_context_->GetKernelExecutor(false)->CreateKernelMod(op_name);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  op_exec_state->kernel = kernel_mod;
  op_exec_state->op_name = op_name;
  op_exec_state->output = output;

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Resize for op " << op_name << " start.";

  auto ret = kernel_mod->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call resize for " << op_name << " failed, error id is " << ret;
  }
  auto work_space = kernel_mod->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    workspace_.max_size = std::max(workspace_.max_size, work_space[0]);
    op_exec_state->workspace = GenerateKernelTensor(kNumberTypeInt8, ShapeVector{SizeToLong(work_space[0])});
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Resize for op " << op_name << " finish.";
}

void SilentChecker::ExecuteCheck(const kernel::KernelMod *kernel_mod, const kernel::KernelTensor *dout,
                                 void *stream_ptr) {
  if (!IsAsdEnable()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(dout);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto iter = check_states_.find(kernel_mod);
  if (iter == check_states_.end()) {
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Not found check state for kernel mod with name "
                                    << kernel_mod->primitive()->name() << ", ignore it.";
    return;
  }
  auto &state = iter->second;

  if (HasApiSilentCheckV3()) {
    LaunchSquareAsync(dout, state, stream_ptr);
    LaunchMaxAsync(dout, state, stream_ptr);
    if (state->is_first_call) {
      state->is_first_call = false;
      LaunchInplaceCopyAsync(dout, state, stream_ptr);
    }
    LaunchSilentCheckV3Async(dout, state, stream_ptr);
  } else {
    LaunchNormAsync(dout, state, stream_ptr);
    LaunchSilentCheckV2Async(dout, state, stream_ptr);
  }
}

void SilentChecker::LaunchOperator(const OpExecState *op_exec_state, const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs, KernelTensor *output_tensor,
                                   void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(op_exec_state->kernel);
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch op " << op_exec_state->op_name << " start.";
  vector<KernelTensor *> workspace;

  if (op_exec_state->output->dev_addr == nullptr) {
    op_exec_state->output->dev_addr = runtime::DeviceAddressUtils::CreateWorkspaceAddress(
      device_context_, kDefaultStreamIndex, op_exec_state->output->max_size);
  }
  if (output_tensor != nullptr) {
    output_tensor->set_device_ptr(op_exec_state->output->dev_addr->GetMutablePtr());
  }

  auto work_space = op_exec_state->kernel->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    if (workspace_.dev_addr == nullptr) {
      workspace_.dev_addr =
        runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, kDefaultStreamIndex, workspace_.max_size);
    }
    op_exec_state->workspace->set_device_ptr(workspace_.dev_addr->GetMutablePtr());
    workspace.emplace_back(op_exec_state->workspace.get());
  }

  if (!op_exec_state->kernel->Launch(inputs, workspace, outputs, stream_ptr)) {
    MS_LOG(EXCEPTION) << "Device do silent check, launch op " << op_exec_state->op_name << " failed.";
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch op " << op_exec_state->op_name << " finish.";
}

void SilentChecker::LaunchNormAsync(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  vector<KernelTensor *> inputs{const_cast<KernelTensor *>(dout), p_scalar_.get(), dim_.get(), keep_dim_.get()};
  vector<KernelTensor *> outputs{state->val.get()};
  LaunchOperator(&state->kernel_norm, inputs, outputs, state->val.get(), stream_ptr);
}

void SilentChecker::LaunchSquareAsync(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  vector<KernelTensor *> inputs{const_cast<KernelTensor *>(dout)};
  vector<KernelTensor *> outputs{state->square.get()};
  LaunchOperator(&state->kernel_square, inputs, outputs, state->square.get(), stream_ptr);
}

void SilentChecker::LaunchMaxAsync(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  vector<KernelTensor *> inputs{state->square.get()};
  vector<KernelTensor *> outputs{state->val.get()};
  LaunchOperator(&state->kernel_max, inputs, outputs, state->val.get(), stream_ptr);
}

void SilentChecker::LaunchInplaceCopyAsync(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  vector<KernelTensor *> inputs{state->avg.get(), state->val.get()};
  vector<KernelTensor *> outputs{};
  LaunchOperator(&state->kernel_copy, inputs, outputs, nullptr, stream_ptr);
}

void SilentChecker::LaunchSilentCheckV2Async(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  vector<KernelTensor *> inputs{state->val.get(),   const_cast<KernelTensor *>(dout),
                                state->sfda.get(),  state->step.get(),
                                c_min_steps_.get(), c_thresh_l1_.get(),
                                c_coeff_l1_.get(),  c_thresh_l2_.get(),
                                c_coeff_l2_.get(),  npu_asd_detect_.get()};
  vector<KernelTensor *> outputs{const_cast<KernelTensor *>(dout), state->sfda.get(), state->step.get(),
                                 state->result.get()};
  LaunchOperator(&state->kernel_silent_check, inputs, outputs, state->result.get(), stream_ptr);
}

void SilentChecker::LaunchSilentCheckV3Async(const KernelTensor *dout, const CheckStatePtr &state, void *stream_ptr) {
  // --------------------------------------------------------------------------------------------------------
  // args_index: | 0   | 1   | 2   | 3          | 4    | 5           | 6           | 7     | 8              |
  // args_name : | val | max | avg | input_grad | step | c_thresh_l1 | c_thresh_l2 | beta1 | npu_asd_detect |
  // --------------------------------------------------------------------------------------------------------
  vector<KernelTensor *> inputs{
    state->val.get(),     state->val.get(),   state->avg.get(),   const_cast<KernelTensor *>(dout),
    state->step.get(),    c_thresh_l1_.get(), c_thresh_l2_.get(), beta1_.get(),
    npu_asd_detect_.get()};
  vector<KernelTensor *> outputs{state->val.get(), const_cast<KernelTensor *>(dout), state->step.get(),
                                 state->result.get()};
  LaunchOperator(&state->kernel_silent_check, inputs, outputs, state->result.get(), stream_ptr);
}

KernelTensorPtr SilentChecker::GenerateKernelTensor(TypeId dtype_id, const ShapeVector &shape, const ValuePtr &value,
                                                    bool alloc_dev) {
  int64_t num_elems = std::accumulate(shape.begin(), shape.end(), 1, [](int64_t x, int64_t y) { return x * y; });
  auto mem_size = UnitSizeInBytes(dtype_id) * num_elems;
  void *addr =
    alloc_dev ? device_context_->device_res_manager_->AllocateMemory(mem_size, kDefaultStreamIndex) : nullptr;
  auto tensor = std::make_shared<kernel::KernelTensor>(addr, mem_size, Format::DEFAULT_FORMAT, dtype_id, shape,
                                                       device_context_->device_context_key().device_name_,
                                                       device_context_->device_context_key().device_id_);
  tensor->set_stream_id(kDefaultStreamIndex);
  tensor->SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape));
  if (value) {
    tensor->SetValue(value);
  }
  return tensor;
}
}  // namespace ascend
}  // namespace silentcheck
}  // namespace mindspore
