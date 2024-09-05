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
#include "backend/common/pass/silent_check_v2.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/param_info.h"
#include "ir/primal_attr.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "kernel/kernel_build_info.h"
#include "mindapi/base/shape_vector.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "op_def/auto_generate/gen_ops_primitive.h"
#include "op_def/framework_ops.h"
#include "infer/l2_normalize.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/info.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kScaleSense[] = "scale_sense";
constexpr char kNpuAsdEnable[] = "NPU_ASD_ENABLE";
constexpr char kParamSfdaPrefix[] = "silent_check_v2.sfda";
constexpr char kParamStepPrefix[] = "silent_check_v2.step";
constexpr char kNameSilentCheckV2[] = "SilentCheckV2";
constexpr int kMinStepDefault = 100;

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
  auto var_val = common::GetEnv(kNpuAsdEnable);
  if (var_val.empty()) {
    return 0;
  }

  if (var_val.size() != 1 || var_val[0] < '0' || var_val[0] > '3') {
    MS_LOG(WARNING) << "Valid values of " << kNpuAsdEnable << " are 0, 1, 2 and 3, but got " << var_val << ".";
    return 0;
  }

  return var_val[0] - '0';
}

bool IsCommOperator(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return false;
  }
  auto prim = GetValuePtr<Primitive>(node);
  return common::AnfAlgo::IsCommunicationOp(prim->name()) && (prim->name() != kBarrierOpName);
}

ValueNodePtr CreateValueNode(const FuncGraphPtr &func_graph, const ValuePtr &value, TypeId dtype,
                             kernel::KernelObjectType obj_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value_node = std::make_shared<ValueNode>(value);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(value->ToAbstract());
  func_graph->AddValueNode(value_node);

  value_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetOutputsFormat({kOpFormat_DEFAULT});
  builder.SetOutputsDeviceType({dtype});
  builder.SetOutputsKernelObjectType({obj_type});
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), value_node.get());

  return value_node;
}
}  // namespace

bool IsNpuAsdEnable() {
  auto ctx = MsContext::GetInstance();
  auto device_target = ctx->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice) {
    return false;
  }
  if (ctx->ascend_soc_version() == kAscendVersion910) {
    return false;
  }
  return GetNpuAsdDetectValue() > 0;
}

using ParamNameValue = std::pair<std::string, tensor::TensorPtr>;
using ParamNameValuePtr = std::shared_ptr<ParamNameValue>;

ParamNameValuePtr GetSfdaParamNameValue(TypeId dtype = kNumberTypeFloat32) {
  static int param_sfda_index = 0;
  constexpr int kSfdaLength = 3;
  // set initial sfda value to 0.0
  float sfda_init[kSfdaLength] = {0.0, 0.0, 0.0};
  return std::make_shared<ParamNameValue>(
    std::pair{std::string(kParamSfdaPrefix) + std::to_string(param_sfda_index++),
              std::make_shared<tensor::Tensor>(dtype, ShapeVector{kSfdaLength}, sfda_init, sizeof(sfda_init))});
}

ParamNameValuePtr GetStepParamNameValue() {
  static int param_step_index = 0;
  constexpr int kStepLength = 1;
  // set initial step values to 0
  int64_t step_init[kStepLength] = {0};
  return std::make_shared<ParamNameValue>(std::pair{
    std::string(kParamStepPrefix) + std::to_string(param_step_index++),
    std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{kStepLength}, step_init, sizeof(step_init))});
}

AnfNodePtr CreateNormForGE(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout) {
  std::vector<AnfNodePtr> square_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameSquare)), dout};
  auto square_node = func_graph->NewCNode(square_inputs);
  MS_EXCEPTION_IF_NULL(square_node);
  square_node->set_abstract(dout->abstract());
  square_node->set_scope(node->scope());

  auto reduce_axes = CreateValueNode(func_graph, std::make_shared<ValueTuple>(std::vector<ValuePtr>{}),
                                     kNumberTypeInt64, kernel::KernelObjectType::TUPLE);
  // set keep_dims and skip_mode to False
  auto false_node =
    CreateValueNode(func_graph, std::make_shared<BoolImm>(false), kNumberTypeBool, kernel::KernelObjectType::SCALAR);
  std::vector<AnfNodePtr> reduce_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameReduceSum)), square_node,
                                           reduce_axes, false_node, false_node};
  auto reduce_node = func_graph->NewCNode(reduce_inputs);
  MS_EXCEPTION_IF_NULL(reduce_node);
  auto ret_abs = dout->abstract()->Clone();
  ret_abs->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{}));
  reduce_node->set_abstract(ret_abs);
  reduce_node->set_scope(node->scope());

  std::vector<AnfNodePtr> sqrt_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameSqrt)), reduce_node};
  auto sqrt_node = func_graph->NewCNode(sqrt_inputs);
  MS_EXCEPTION_IF_NULL(sqrt_node);
  sqrt_node->set_abstract(reduce_node->abstract());
  sqrt_node->set_scope(node->scope());

  return sqrt_node;
}

AnfNodePtr CreateNormForKBK(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout) {
  auto ord =
    CreateValueNode(func_graph, std::make_shared<FP32Imm>(2), kNumberTypeFloat32, kernel::KernelObjectType::SCALAR);
  auto dims = CreateValueNode(func_graph, std::make_shared<ValueTuple>(std::vector<ValuePtr>{}), kNumberTypeInt64,
                              kernel::KernelObjectType::TUPLE);
  auto keep_dims =
    CreateValueNode(func_graph, std::make_shared<BoolImm>(false), kNumberTypeBool, kernel::KernelObjectType::SCALAR);
  std::vector<AnfNodePtr> norm_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameNorm)), dout, ord, dims,
                                         keep_dims};
  auto norm_node = func_graph->NewCNode(norm_inputs);
  MS_EXCEPTION_IF_NULL(norm_node);
  auto abs_tensor = dout->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(abs_tensor);
  auto norm_abs = abs_tensor->abstract::AbstractTensor::Clone();
  norm_abs->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{}));
  norm_node->set_abstract(norm_abs);
  norm_node->set_scope(node->scope());

  return norm_node;
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, TypeId dst_type) {
  auto input_abs = input->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_abs);
  auto src_type = input_abs->element()->GetType()->type_id();
  if (src_type == dst_type) {
    return input;
  }

  PrimitivePtr cast_prim = std::make_shared<Primitive>(prim::kPrimCast->name());
  MS_EXCEPTION_IF_NULL(cast_prim);
  (void)cast_prim->AddAttr("dst_type", TypeIdToType(dst_type));
  (void)cast_prim->AddAttr("DstT", TypeIdToType(dst_type));
  (void)cast_prim->AddAttr("SrcT", TypeIdToType(src_type));
  // Create dest type node.
  auto dst_type_ptr = TypeIdToType(dst_type);
  auto dst_type_node = CreateValueNode(func_graph, std::make_shared<Int64Imm>(dst_type), kNumberTypeInt64,
                                       kernel::KernelObjectType::SCALAR);

  // Insert Cast node
  std::vector<AnfNodePtr> cast_inputs = {NewValueNode(cast_prim), input, dst_type_node};
  auto cast_node = func_graph->NewCNode(cast_inputs);
  MS_EXCEPTION_IF_NULL(cast_node);
  auto cast_abs = input_abs->Clone()->cast<abstract::AbstractTensorPtr>();
  cast_abs->element()->set_type(dst_type_ptr);
  MS_EXCEPTION_IF_NULL(cast_abs);
  cast_node->set_abstract(cast_abs);

  return cast_node;
}

AnfNodePtr GetGradValue(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &dout,
                        const ParameterPtr &loss_scale) {
  if (loss_scale == nullptr) {
    return dout;
  }

  auto umonad_node = NewValueNode(std::make_shared<UMonad>());
  umonad_node->set_abstract(std::make_shared<abstract::AbstractUMonad>());
  std::vector<AnfNodePtr> load_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimLoad->name())), loss_scale,
                                         umonad_node};
  auto load_node = func_graph->NewCNode(load_inputs);
  MS_EXCEPTION_IF_NULL(load_node);
  auto scale_param_abs = loss_scale->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(scale_param_abs);
  load_node->set_abstract(scale_param_abs->abstract::AbstractTensor::Clone());
  load_node->set_scope(node->scope());

  auto dout_abs = dout->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(dout_abs);
  auto dst_type = dout_abs->element()->GetType()->type_id();
  // Ascend Div operator does not support bf16, in this case select fp32 as the middle computing data type
  auto compute_type = (dst_type == kNumberTypeBFloat16 ? kNumberTypeFloat32 : dst_type);
  // create cast node if the type of scale_sense is not the same as type of dout
  auto cast_dout = CreateCastNode(func_graph, dout, compute_type);
  auto cast_scale = CreateCastNode(func_graph, load_node, compute_type);
  std::vector<AnfNodePtr> div_inputs = {NewValueNode(std::make_shared<Primitive>(ops::kNameDiv)), cast_dout,
                                        cast_scale};
  auto div_node = func_graph->NewCNode(div_inputs);
  MS_EXCEPTION_IF_NULL(div_node);
  div_node->set_abstract(cast_dout->abstract());
  div_node->set_scope(node->scope());

  return CreateCastNode(func_graph, div_node, dst_type);
}

void SilentCheckV2::GetLossScale() {
  MS_EXCEPTION_IF_NULL(root_);
  auto parameters = root_->parameters();
  for (const auto &param : parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    const auto &name = param_ptr->name();
    if (name == kScaleSense) {
      loss_scale_ = param_ptr;
    }
  }
}

AnfNodePtr SilentCheckV2::CreateSlientCheckNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // skip forward node in graph
  if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    return node;
  }

  MS_LOG(INFO) << cnode->fullname_with_scope() << " has attr forward_unique_id=" << std::boolalpha
               << GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));

  // create SlientCheckV2 node
  auto check_prim = std::make_shared<Primitive>(kNameSilentCheckV2);
  check_prim->AddAttr("side_effect_mem", std::make_shared<BoolImm>(true));
  // input1: input_grad
  auto dout = GetGradValue(func_graph, node, cnode->input(kIndexOne), loss_scale_);
  // input0: val
  auto norm_node = MsContext::GetInstance()->GetJitLevel() == kAttrJitLevelO2
                     ? CreateNormForGE(func_graph, node, dout)
                     : CreateNormForKBK(func_graph, node, dout);
  // input2: sfda
  auto param_sfda =
    CreateValueNode(func_graph, GetSfdaParamNameValue()->second, kNumberTypeFloat32, kernel::KernelObjectType::TENSOR);
  // input3: step
  auto param_step =
    CreateValueNode(func_graph, GetStepParamNameValue()->second, kNumberTypeInt64, kernel::KernelObjectType::TENSOR);
  // input4: cMinSteps
  auto min_steps = CreateValueNode(func_graph, std::make_shared<Int64Imm>(kMinStepDefault), kNumberTypeInt64,
                                   kernel::KernelObjectType::SCALAR);
  // input5: cThreshL1
  auto upper_thresh = parse_thresh("NPU_ASD_UPPER_THRESH", "1000000,10000", 3);
  auto thresh_l1 = CreateValueNode(func_graph, std::make_shared<FP32Imm>(upper_thresh.front()), kNumberTypeFloat32,
                                   kernel::KernelObjectType::SCALAR);
  // input7: cThreshL2
  auto thresh_l2 = CreateValueNode(func_graph, std::make_shared<FP32Imm>(upper_thresh.back()), kNumberTypeFloat32,
                                   kernel::KernelObjectType::SCALAR);
  // input6: cCoeffL1
  auto sigma_thresh = parse_thresh("NPU_ASD_SIGMA_THRESH", "100000,5000", 3);
  auto coeff_l1 = CreateValueNode(func_graph, std::make_shared<FP32Imm>(sigma_thresh.front()), kNumberTypeFloat32,
                                  kernel::KernelObjectType::SCALAR);
  // input8: cCoeffL2
  auto coeff_l2 = CreateValueNode(func_graph, std::make_shared<FP32Imm>(sigma_thresh.back()), kNumberTypeFloat32,
                                  kernel::KernelObjectType::SCALAR);
  // input9: npuAsdDetect
  auto npu_asd_detect = CreateValueNode(func_graph, std::make_shared<Int64Imm>(GetNpuAsdDetectValue()),
                                        kNumberTypeInt64, kernel::KernelObjectType::SCALAR);
  std::vector<AnfNodePtr> check_inputs = {NewValueNode(check_prim),
                                          norm_node,
                                          dout,
                                          param_sfda,
                                          param_step,
                                          min_steps,
                                          thresh_l1,
                                          coeff_l1,
                                          thresh_l2,
                                          coeff_l2,
                                          npu_asd_detect};
  auto check_node = func_graph->NewCNode(check_inputs);
  MS_EXCEPTION_IF_NULL(check_node);
  auto tensor_abs = dout->abstract()->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_abs);
  // output0: input_grad
  auto out_input_grad_abs = tensor_abs->abstract::AbstractTensor::Clone();
  // output1: sfda
  auto out_sfda_abs = param_sfda->abstract()->cast<abstract::AbstractTensorPtr>()->abstract::AbstractTensor::Clone();
  // output2: step
  auto out_step_abs = param_step->abstract()->cast<abstract::AbstractTensorPtr>()->abstract::AbstractTensor::Clone();
  // output3: result
  auto out_result_abs = std::make_shared<abstract::AbstractTensor>(kInt32, ShapeVector{});
  check_node->set_abstract(std::make_shared<abstract::AbstractTuple>(
    AbstractBasePtrList{out_input_grad_abs, out_sfda_abs, out_step_abs, out_result_abs}));
  check_node->set_scope(node->scope());

  // create Depend node
  std::vector<AnfNodePtr> depend_inputs = {NewValueNode(std::make_shared<Primitive>(kDependOpName)),
                                           cnode->input(kIndexOne), check_node};
  auto depend_node = func_graph->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(dout->abstract());
  depend_node->set_scope(node->scope());
  return depend_node;
}

bool SilentCheckV2::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  bool changed = false;

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    // skip forward node in graph
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    // skip non-communicator operators
    if (!IsCommOperator(cnode->input(ops::kInputIndex0))) {
      continue;
    }
    auto check_node = CreateSlientCheckNode(func_graph, node);
    // update cnode input
    cnode->set_input(ops::kInputIndex1, check_node);
    manager->SetEdge(cnode, ops::kInputIndex1, check_node);
    changed = true;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
