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
#include "pipeline/jit/ps/silent_check.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <queue>
#include <map>
#include <set>
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
#include "ir/core_ops_name.h"
#include "ir/dtype/number.h"
#include "ir/func_graph.h"
#include "ir/param_info.h"
#include "ir/primal_attr.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "common/kernel_build_info.h"
#include "mindapi/base/shape_vector.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "op_def/auto_generate/gen_ops_primitive.h"
#include "op_def/framework_ops.h"
#include "op_def/structure_ops.h"
#include "op_def/other_ops.h"
#include "infer/l2_normalize.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/info.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "availability/silent_check/silent_check.h"

namespace mindspore {
namespace pipeline {
namespace {
constexpr char kScaleSense[] = "scale_sense";
constexpr char kNpuAsdEnable[] = "NPU_ASD_ENABLE";
constexpr char kParamSfdaPrefix[] = "silent_check_v2.sfda";
constexpr char kParamStepPrefix[] = "silent_check_v2.step";
constexpr char kNameSilentCheckV2[] = "SilentCheckV2";
constexpr int kMinStepDefault = 100;

bool NeedCheckCommOperator(const AnfNodePtr &node) {
  if (!IsValueNode<Primitive>(node)) {
    return false;
  }
  // skip barrier and receive operator
  if (IsOneOfPrimitive(node, {prim::kPrimBarrier, prim::kPrimReceive})) {
    return false;
  }
  auto prim = GetValuePtr<Primitive>(node);
  if (!common::AnfAlgo::IsCommunicationOp(prim->name())) {
    return false;
  }
  return prim->instance_name().find(parallel::REDISTRIBUTION_OP) != std::string::npos ||
         prim->instance_name().find(parallel::FORWARD_OP) != std::string::npos;
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

ParameterPtr GetScaleSense(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto parameters = func_graph->parameters();
  for (const auto &param : parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (param_ptr->name() == kScaleSense) {
      return param_ptr;
    }
  }
  return nullptr;
}

bool HasFloat16Type(const abstract::AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  if (abs_base->isa<abstract::AbstractScalar>()) {
    return abs_base->cast<abstract::AbstractScalarPtr>()->GetType()->type_id() == kNumberTypeFloat16;
  }
  if (abs_base->isa<abstract::AbstractTensor>()) {
    return abs_base->cast<abstract::AbstractTensorPtr>()->element()->GetType()->type_id() == kNumberTypeFloat16;
  }
  if (abs_base->isa<abstract::AbstractSequence>()) {
    auto abs_seq = abs_base->cast<abstract::AbstractSequencePtr>();
    if (abs_seq->dynamic_len()) {
      return HasFloat16Type(abs_seq->dynamic_len_element_abs());
    }
    for (size_t i = 0; i < abs_seq->size(); ++i) {
      if (HasFloat16Type((*abs_seq)[i])) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

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

void SilentCheckV2::GetLossScale() { loss_scale_ = GetScaleSense(root_); }

bool SilentCheckV2::HasFloat16Input() {
  auto iter = std::find_if(root_->parameters().begin(), root_->parameters().end(),
                           [](const AnfNodePtr &param) { return HasFloat16Type(param->abstract()); });
  if (iter != root_->parameters().end()) {
    MS_LOG(WARNING) << "Graph " << root_->ToString() << " has parameter " << (*iter)->ToString() << " with type "
                    << (*iter)->abstract()->ToString() << ", skip inserting silent check operators.";
    return true;
  }

  // check whether the output type of GetNext node contains float16 type
  auto node_get_next = FindGetNextNode();
  if (node_get_next != nullptr && HasFloat16Type(node_get_next->abstract())) {
    MS_LOG(WARNING) << "GetNext node of graph " << root_->ToString() << " with output type "
                    << node_get_next->abstract()->ToString() << ", skip inserting silent check operators.";
    return true;
  }

  return false;
}

CNodePtr SilentCheckV2::GetLastGradNode(const FuncGraphPtr &func_graph, const AnfNodePtr &start_node) {
  auto manager = func_graph->manager();

  // map's key: forward_unique_id, value: CNode in backward graph
  std::map<std::string, CNodePtr> grad_node_map;
  for (auto &node : manager->all_nodes()) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    auto forward_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));
    grad_node_map.emplace(std::make_pair(forward_unique_id, cnode));
  }
  MS_LOG(INFO) << "grad_node_map.size=" << grad_node_map.size();

  auto &node_users = manager->node_users();
  std::queue<AnfNodePtr> candidates;
  std::set<AnfNodePtr> visited;
  auto push_candidate_node = [&candidates, &visited](const AnfNodePtr &cand_node) {
    if (visited.count(cand_node)) {
      return;
    }
    candidates.push(cand_node);
    visited.insert(cand_node);
  };
  push_candidate_node(start_node);
  while (!candidates.empty()) {
    auto node = candidates.front();
    candidates.pop();
    MS_LOG(DEBUG) << node->DebugString();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      continue;
    }
    for (auto &elem : iter->second) {
      auto &user_node = elem.first;
      auto cnode = user_node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      if (visited.count(user_node)) {
        continue;
      }
      if (cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
        auto node_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrUniqueId));
        if (grad_node_map.count(node_unique_id)) {
          auto grad_node = grad_node_map[node_unique_id]->cast<CNodePtr>();
          // normally dout is a tensor, so here expect grad-node input1 is a tensor
          if (grad_node != nullptr && grad_node->size() > kIndex1 &&
              grad_node->input(kIndex1)->abstract()->isa<abstract::AbstractTensor>()) {
            MS_LOG(INFO) << "Found grad node " << grad_node->DebugString()
                         << " based on start node: " << start_node->DebugString() << " for graph "
                         << func_graph->ToString();
            return grad_node;
          }
        }
      }
      if (common::AnfAlgo::IsCallNode(cnode)) {
        auto op = cnode->input(kIndex0);
        MS_EXCEPTION_IF_NULL(op);
        if (IsValueNode<FuncGraph>(op)) {
          FuncGraphPtr fg = GetValueNode<FuncGraphPtr>(op);
          auto param = fg->parameters()[elem.second - 1]->cast<ParameterPtr>();
          MS_LOG(INFO) << "Encounter func_graph: " << fg->ToString() << " user_node param_index=" << elem.second << "/"
                       << fg->parameters().size() << " param_name=" << param->name();
          push_candidate_node(param);
        } else {
          push_candidate_node(user_node);
        }
      } else {
        push_candidate_node(user_node);
      }
    }
  }

  MS_LOG(INFO) << "Not found grad node based on start node: " << start_node->DebugString() << " for graph "
               << func_graph->ToString();
  return nullptr;
}

void SilentCheckV2::GetLastGradNode() {
  MS_EXCEPTION_IF_NULL(root_);
  auto parameters = root_->parameters();
  for (const auto &param : parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    // skip network weight which has default value
    if (param_ptr->has_default()) {
      continue;
    }
    MS_LOG(INFO) << "Consider param " << param_ptr->name() << " as start node to find last grad node";
    auto grad_node = GetLastGradNode(root_, param);
    if (grad_node != nullptr) {
      last_grad_node_ = grad_node;
      return;
    }
  }

  auto get_next_node = FindGetNextNode();
  if (get_next_node != nullptr) {
    MS_LOG(INFO) << "GetNext node is " << get_next_node->DebugString() << " of graph "
                 << get_next_node->func_graph()->ToString();
    auto grad_node = GetLastGradNode(root_, get_next_node);
    if (grad_node != nullptr) {
      last_grad_node_ = grad_node;
      return;
    }
  }

  MS_LOG(INFO) << "Not found suitable grad node for root graph " << root_->ToString();
}

bool SilentCheckV2::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);

  auto fn_mark_node_need_check = [this](AnfNodePtrList &nodes) -> bool {
    bool changed = false;
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      // building graph node users when the network contains loss-scale, since
      // parameter `scale_sense` may be added to subgraph
      if (loss_scale_ != nullptr) {
        for (const auto &input : cnode->inputs()) {
          if (IsValueNode<FuncGraph>(input)) {
            auto sub_graph = GetValueNode<FuncGraphPtr>(input);
            graph_users_[sub_graph].insert(cnode);
          }
        }
      }
      // skip forward node in graph
      if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
        continue;
      }
      // skip communicator operators which don't need check
      if (!((cnode == last_grad_node_) || NeedCheckCommOperator(cnode->input(ops::kInputIndex0)))) {
        continue;
      }
      // add attriute "need_silent_check" to cnode
      auto silent_check_op_type = common::AnfAlgo::IsCommunicationOp(cnode) ? silentcheck::kSilentCheckGradCommOp
                                                                            : silentcheck::kSilentCheckGradLastOp;
      cnode->AddPrimalAttr(silentcheck::kAttrSilentCheckOpType, MakeValue<int>(silent_check_op_type));
    }
    return changed;
  };

  if (func_graph == root_) {
    return fn_mark_node_need_check(GetRootGraphTopoNodes());
  } else {
    auto graph_nodes = TopoSort(func_graph->get_return());
    return fn_mark_node_need_check(graph_nodes);
  }
}

AnfNodePtr SilentCheckV2::FindGetNextNode() {
  for (auto &node : GetRootGraphTopoNodes()) {
    if (IsPrimitiveCNode(node, prim::kPrimGetNext)) {
      return node;
    }
  }

  return nullptr;
}

AnfNodePtrList &SilentCheckV2::GetRootGraphTopoNodes() {
  MS_EXCEPTION_IF_NULL(root_);
  if (root_graph_nodes_.empty()) {
    root_graph_nodes_ = TopoSort(root_->return_node());
  }
  return root_graph_nodes_;
}

bool SilentCheckPass(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  FuncGraphPtr root_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(root_graph);

  auto silent_check = std::make_shared<SilentCheckV2>(root_graph);
  // skip inserting silent check operator if root graph has fp16 input
  if (silent_check->HasFloat16Input()) {
    return true;
  }
  // find last grad node in graphs
  silent_check->GetLastGradNode();
  // insert silent check operator for root graph
  silent_check->Run(root_graph);

  // insert silent check operator for sub-graphs
  auto manager = root_graph->manager();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(manager);
  const auto &sub_graphs = manager->func_graphs_used_total(root_graph);
  for (const auto &sub_graph : sub_graphs) {
    silent_check->Run(sub_graph);
  }

  return true;
}
}  // namespace pipeline
}  // namespace mindspore
