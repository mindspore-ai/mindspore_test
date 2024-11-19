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

#include "kernel/ascend/availability/silent_check/dynamic_graph_check.h"
#include <sys/param.h>
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
#include "kernel/common/pyboost/auto_generate/masked_select.h"
#include "kernel/common/pyboost/auto_generate/max.h"
#include "kernel/common/pyboost/auto_generate/median_ext.h"
#include "kernel/common/pyboost/auto_generate/ne_scalar.h"
#include "kernel/common/pyboost/auto_generate/norm.h"
#include "kernel/common/pyboost/auto_generate/silent_check_v2.h"
#include "kernel/common/pyboost/auto_generate/silent_check_v3.h"
#include "kernel/common/pyboost/auto_generate/square.h"
#include "kernel/common/pyboost/op_register.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "op_def/op_name.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace silentcheck {
namespace ascend {
using mindspore::kernel::pyboost::MaskedSelect;
using mindspore::kernel::pyboost::MedianExt;
using mindspore::kernel::pyboost::MemBlock;
using mindspore::kernel::pyboost::NeScalar;
using mindspore::kernel::pyboost::Norm;
using mindspore::kernel::pyboost::OpRunner;
using mindspore::kernel::pyboost::PyBoostUtils;
using mindspore::kernel::pyboost::SilentCheckV2;
using mindspore::kernel::pyboost::Square;

namespace {
constexpr char kNpuAsdEnable[] = "NPU_ASD_ENABLE";
constexpr char kNameSilentCheckV2[] = "aclnnSilentCheckV2";
constexpr char kNameSilentCheckV3[] = "aclnnSilentCheckV3";
constexpr float kSilentCheckV3Beta = 0.99;
constexpr char kUpperThreshDefaultVal[] = "1000000,10000";
constexpr char kSigmaThreshDefaultVal[] = "100000,5000";
constexpr float kThreshMinimalVal = 3;

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
      MS_LOG(WARNING) << "Valid values of " << kNpuAsdEnable << " are 0, 1, 2 and 3, but got " << var_val << ".";
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
}  // namespace

bool DynamicSilentChecker::IsNpuAsdEnable() {
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

CheckObject::CheckObject() {
  if (HasApiSilentCheckV3()) {
    square_op_ = CREATE_PYBOOST_OP(Square, kAscendDevice);
    max_op_ = CREATE_PYBOOST_OP(Max, kAscendDevice);
    silent_check_v3_op_ = CREATE_PYBOOST_OP(SilentCheckV3, kAscendDevice);

    ne_scalar_op_ = CREATE_PYBOOST_OP(NeScalar, kAscendDevice);
    masked_select_op_ = CREATE_PYBOOST_OP(MaskedSelect, kAscendDevice);
    median_op_ = CREATE_PYBOOST_OP(MedianExt, kAscendDevice);
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
  if (state->is_first_call) {
    state->is_first_call = false;
    LaunchNeScalar();
    LaunchMaskedSelect();
    LaunchMedian(state);
  }
  LaunchMax();
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
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void CheckObject::LaunchSilentCheckV2(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  auto &op = silent_check_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  // create value used by aclnnSilentCheck
  auto val = norm_op_->outputs()[kIndex0];
  auto sfda = state->sfda;
  auto step = state->step;
  auto upper_thresh = parse_thresh("NPU_ASD_UPPER_THRESH", kUpperThreshDefaultVal, kThreshMinimalVal);
  auto sigma_thresh = parse_thresh("NPU_ASD_SIGMA_THRESH", kSigmaThreshDefaultVal, kThreshMinimalVal);
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
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void CheckObject::LaunchSilentCheckV3(const BaseTensorPtr &input_grad, const DynamicCheckStatePtr &state) {
  auto &op = silent_check_v3_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  max_op_->outputs()[0]->set_shape(ShapeVector{1});

  auto val = max_op_->outputs()[0];
  auto max = max_op_->outputs()[0];
  auto avg = state->avg;
  // input_grad
  auto step = state->step;
  auto dst_size =
    std::make_shared<BaseTensor>(kNumberTypeInt64, ShapeVector{static_cast<int64_t>(input_grad->shape_c().size())},
                                 input_grad->shape_c().data(), input_grad->shape_c().size() * sizeof(int64_t));
  auto dst_stride =
    std::make_shared<BaseTensor>(kNumberTypeInt64, ShapeVector{static_cast<int64_t>(input_grad->stride().size())},
                                 input_grad->shape_c().data(), input_grad->stride().size() * sizeof(int64_t));
  int64_t offset = input_grad->storage_offset();
  auto dst_offset = std::make_shared<BaseTensor>(kNumberTypeInt64, ShapeVector{1}, &offset, sizeof(offset));

  auto upper_thresh = parse_thresh("NPU_ASD_UPPER_THRESH", "1000000,10000", 3);
  auto c_thresh_l1_ptr = std::make_shared<FP32Imm>(upper_thresh.front());
  auto c_thresh_l2_ptr = std::make_shared<FP32Imm>(upper_thresh.back());
  auto beta_ptr = std::make_shared<FP32Imm>(kSilentCheckV3Beta);
  auto npu_asd_detect_ptr = std::make_shared<Int64Imm>(GetNpuAsdDetectValue());

  OpRunner::InferOpOutput(op, val, max, avg, input_grad, step, dst_size, dst_stride, dst_offset, c_thresh_l1_ptr,
                          c_thresh_l2_ptr, beta_ptr, npu_asd_detect_ptr);

  auto c_thresh_l1 = GetValue<pyfloat>(c_thresh_l1_ptr);
  auto c_thresh_l2 = GetValue<pyfloat>(c_thresh_l2_ptr);
  auto beta = GetValue<pyfloat>(beta_ptr);
  auto npu_asd_detect = GetValue<int64_t>(npu_asd_detect_ptr);

  // op->set_outputs(std::vector<tensor::BaseTensorPtr>{avg, input_grad, step, op->output(kIndex3)});
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), val, max, avg, input_grad, step, dst_size,
                                dst_stride, dst_offset);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(),
                                 std::vector<BaseTensorPtr>{op->output(kIndex3)});

  auto device_context = op->device_context();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(op->device_context(), val, max, avg, input_grad, step, dst_size, dst_stride, dst_offset);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(op->device_context(), std::vector<BaseTensorPtr>{op->output(kIndex3)});
  LAUNCH_ACLNN(aclnnSilentCheckV3, device_context, op->stream_id(), val, max, avg, input_grad, step, dst_size,
               dst_stride, dst_offset, c_thresh_l1, c_thresh_l2, beta, npu_asd_detect, op->output(kIndex3));

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
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

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void CheckObject::LaunchNeScalar() {
  auto &op = ne_scalar_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto &input_tensor = square_op_->outputs()[kIndex0];
  ScalarPtr other = std::make_shared<Int8Imm>(0);

  OpRunner::InferOpOutput(op, input_tensor, other);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
  // Malloc for output tensors
  const auto &outputs = op->outputs();
  PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
  LAUNCH_ACLNN(aclnnNeScalar, device_context, op->stream_id(), input_tensor, other, outputs[0]);

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void CheckObject::LaunchMaskedSelect() {
  auto &op = masked_select_op_;

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto &input_tensor = square_op_->outputs()[kIndex0];
  auto &mask_tensor = ne_scalar_op_->outputs()[kIndex0];

  auto device_context = op->device_context();
  auto stream_id = op->stream_id();
  OpRunner::InferOpOutput(op, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tensor, mask_tensor);
  PyBoostUtils::PrepareOpOutputs(device_context, stream_id, op->outputs());

  runtime::Pipeline::Get().WaitForward();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor, mask_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);
  auto return_value =
    LAUNCH_ACLNN_SYNC(aclnnMaskedSelect, device_context, op->stream_id(), input_tensor, mask_tensor, outputs[0]);
  const auto &cache_func_ptr = std::get<kIndex2>(return_value);
  auto all_acl_tensor = cache_func_ptr(transform::ProcessCacheType::kGetOutputShape, {});

  auto output_real_shape = all_acl_tensor[kIndex2];
  auto simple_infer_ptr = op->output_value_simple_info();
  simple_infer_ptr->shape_vector_ = ShapeArray{output_real_shape};

  op->UpdateOutputShape(op->output(kIndex0), output_real_shape);
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void CheckObject::LaunchMedian(const DynamicCheckStatePtr &state) {
  auto &op = median_op_;
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Call " << op->primitive()->name() << " start";

  auto &input_tensor = masked_select_op_->outputs()[kIndex0];
  op->set_outputs(std::vector<tensor::BaseTensorPtr>{state->avg});

  OpRunner::InferOpOutput(op, input_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto device_context = op->device_context();
  const auto &outputs = op->outputs();
  // Malloc for input tensors
  PyBoostUtils::MallocOpInputs(device_context, input_tensor);
  // Malloc for output tensors
  PyBoostUtils::MallocOpOutputs(device_context, outputs);

  LAUNCH_ACLNN(aclnnMedian, device_context, op->stream_id(), input_tensor, outputs[0]);

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
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

  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Launch " << op->primitive()->name() << " end";
}

void DynamicSilentChecker::DoSilentCheck(const std::string &op_name, const std::string &comm_group,
                                         const BaseTensorPtr &input_grad) {
  static bool is_npu_asd_enable = IsNpuAsdEnable();
  if (!is_npu_asd_enable) {
    return;
  }
  if (!is_back_prop_) {
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
}  // namespace ascend
}  // namespace silentcheck
}  // namespace mindspore
