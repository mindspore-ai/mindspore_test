/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/pynative_jit_grad.h"

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <set>
#include "backend/graph_compiler/transform.h"
#include "pynative/pynative_utils.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/pynative/common_utils.h"
#include "pipeline/jit/ps/pass.h"
#include "ir/func_graph_cloner.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "frontend/optimizer/irpass/check_invalid_view_inplace_dout.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_v.h"
namespace mindspore {
namespace ad {
mindspore::HashMap<std::string, std::pair<FuncGraphPtr, FuncGraphPtr>> pass_grad_graph_;
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_forward_resource;
mindspore::HashMap<std::string, FuncGraphPtr> original_bprop_graph;
std::set<std::string> check_invalid_dout_bprop_graph;

namespace {
static const std::vector<PrimitivePtr> UNREUSED_PRIM_LIST = {
  prim::kPrimStopGradient, prim::kPrimUpdateState,      prim::kPrimMirror,
  prim::kPrimVirtualDiv,   prim::kPrimMutable,          prim::kPrimInsertGradientOf,
  prim::kPrimHookBackward, prim::kPrimCellBackwardHook, prim::kPrimPrintShapeType};

// Optimizes the forward function graph.
FuncGraphPtr OptimizeForwardGraph(const FuncGraphPtr &bprop_func_graph, bool need_renormalize = false) {
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  if (need_renormalize) {
    // Renormalize, infer shape and set abstract for all nodes in graph
    abstract::AbstractBasePtrList args_abs;
    const auto &parameters = bprop_func_graph->parameters();
    (void)std::transform(parameters.begin(), parameters.end(), std::back_inserter(args_abs),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    MS_LOG(INFO) << "Start renormalizing for graph: " << bprop_func_graph->ToString();
    FuncGraphPtr new_fg = pipeline::Renormalize(resource, bprop_func_graph, args_abs);
    MS_EXCEPTION_IF_NULL(new_fg);
    MS_LOG(INFO) << "Finish renormalizing for graph: " << bprop_func_graph->ToString();
    resource->set_func_graph(new_fg);
    resource->set_args_abs(args_abs);
    manager->AddFuncGraph(new_fg);
  }
  (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
  (void)OptAfterJitGradPass(resource);
  return resource->func_graph();
}

// Optimizes the bprop function graph using certain passes
FuncGraphPtr OptimizeBpropGraph(const FuncGraphPtr &bprop_func_graph, const pynative::GradParamPtr &grad_param) {
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_func_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_func_graph);
  auto after_opt_bg = pipeline::JitBpropGraphPass(resource, true);
  auto is_dynamic_shape_control_flow = grad_param->is_jit_graph && grad_param->is_control_flow;
  if (is_dynamic_shape_control_flow) {
    for (const auto &g : manager->func_graphs()) {
      g->set_flag(kFlagJitCallGraph, true);
    }
  }
  return after_opt_bg;
}

void ClearFuncGraphCNodeAbstract(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto &node : nodes) {
    if (node == nullptr || node->isa<Parameter>() || node->isa<mindspore::ValueNode>()) {
      continue;
    }
    const AbstractBasePtr &prev_inferred = node->abstract();
    // Keep previous inferred value for parameter and ValueNode if the inferred value is not AbstractFunction.
    if (prev_inferred != nullptr && prev_inferred->isa<abstract::AbstractFunction>()) {
      continue;
    }
    node->set_abstract(nullptr);
    MS_LOG(DEBUG) << "Abstract of node " << node->DebugString() << " is set to nullptr";
  }
}

void PlantFuncGradBpropGraphDout(const FuncGraphPtr &graph, size_t dout_index,
                                 const abstract::AbstractBasePtr &out_abstract) {
  MS_EXCEPTION_IF_NULL(graph);
  if (const size_t param_size = graph->parameters().size(); param_size <= dout_index) {
    MS_LOG(EXCEPTION) << "Invalid dout index for bprop_func_graph: " << graph->ToString()
                      << " , total param size: " << param_size << " , dout_index: " << dout_index;
  }
  // Plant dout tuple or dict
  // Parameters for bprop graph: {original_inputs, dout}
  if (out_abstract->isa<abstract::AbstractSequence>()) {
    pynative::CommonUtils::ProcessTupleParam(graph, dout_index);
  } else if (out_abstract->isa<abstract::AbstractDictionary>()) {
    pynative::CommonUtils::ProcessDictParam(graph, dout_index);
  }
}

bool IsUnSupportPrim(const AnfNodePtr &node) {
  // Check if a cnode
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return true;
  }
  // Check if a prim cnode
  const auto &input = cnode->input(0);
  MS_EXCEPTION_IF_NULL(input);
  if (!GetValuePtr<Primitive>(input)) {
    return true;
  }
  // Filter unsupported prim
  return std::any_of(UNREUSED_PRIM_LIST.begin(), UNREUSED_PRIM_LIST.end(),
                     [&node](const auto &primitive) { return IsPrimitiveCNode(node, primitive); });
}

bool JitOutputHasDict(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractDictionary>()) {
    return true;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_sequence = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_sequence->elements().begin(), abs_sequence->elements().end(),
                       [](const abstract::AbstractBasePtr &item) { return JitOutputHasDict(item); });
  }
  return false;
}

BaseRef GetGraphResult(const FuncGraphPtr &fg, const VectorRef &arg_list, bool cache_hit,
                       const std::string &cache_key) {
  pipeline::ResourcePtr resource;
  const auto &it = jit_forward_resource.find(cache_key);
  if (it == jit_forward_resource.end()) {
    if (cache_hit) {
      MS_LOG(WARNING) << "Can not find cached resource for func graph: " << fg->ToString();
    }
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(fg);
    auto manager = resource->manager();
    manager->AddFuncGraph(resource->func_graph(), true);
    (void)TaskEmitAction(resource);
    (void)ExecuteAction(resource);
    jit_forward_resource[cache_key] = resource;
  } else {
    resource = it->second;
  }
  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  auto result = (*run)(arg_list);
  MS_LOG(INFO) << "Finish running funcgraph: " << fg->ToString() << " , result: " << result.ToString();
  return result;
}

AnfNodePtrList ProcessParam(const FuncGraphPtr &source_fg, const abstract::AbstractBasePtrList &input_abs,
                            const std::vector<ValuePtr> &input_values) {
  MS_EXCEPTION_IF_NULL(source_fg);
  AnfNodePtrList param_list;
  if (input_abs.size() != input_values.size()) {
    MS_LOG(EXCEPTION) << "Got unmatched input abstract and value.";
  }
  for (size_t index = 0; index < input_abs.size(); ++index) {
    auto param = source_fg->add_parameter();
    param->set_abstract(input_abs[index]);
    (void)param_list.emplace_back(param);
    const auto &input_value = input_values[index];
    MS_EXCEPTION_IF_NULL(input_value);
    if (!input_value->isa<tensor::Tensor>()) {
      continue;
    }
    const auto &tensor = input_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    if (!tensor->is_parameter()) {
      continue;
    }
    const auto &param_info = tensor->param_info();
    if (param_info) {
      const auto &parameter = param_info->parameter();
      if (parameter && parameter->has_default()) {
        param->set_default_param(parameter->default_param_raw());
      }
    }
  }
  return param_list;
}

ValuePtr PyObjToValue(const py::object &obj, bool stub = false) {
  ValuePtr converted_ret;
  if (stub) {
    converted_ret = parse::data_converter::PyDataToStubNode(obj);
  } else {
    converted_ret = parse::data_converter::PyDataToValue(obj);
  }
  if (converted_ret == nullptr) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << ConvertPyObjToString(obj);
  }
  return converted_ret;
}

// Helper function to handle forward result
py::object HandleForwardResult(const BaseRef &forward_result, const FuncGraphPtr &forward_fg,
                               const AbstractBasePtr &origin_forward_output_abs,
                               const pynative::GradParamPtr &grad_param, bool need_reuse_forward_node) {
  MS_EXCEPTION_IF_NULL(forward_result);
  MS_EXCEPTION_IF_NULL(forward_fg);
  if (!need_reuse_forward_node) {
    return pipeline::BaseRefToPyDataWithUserData(forward_result, origin_forward_output_abs);
  }
  grad_param->added_args.clear();
  if (utils::isa<VectorRef>(forward_result)) {
    MS_LOG(INFO) << "Run forward graph: " << forward_fg->ToString() << " in sync pipeline mode.";
    auto vector_result = utils::cast<VectorRef>(forward_result);
    auto result = vector_result[kIndex0];
    VectorRef add_args(vector_result.begin() + 1, vector_result.end());
    grad_param->added_args = add_args;
    return pipeline::BaseRefToPyDataWithUserData(result, origin_forward_output_abs);
  } else {
    MS_LOG(INFO) << "Run forward graph: " << forward_fg->ToString() << " in async pipeline mode.";
    const auto &output = forward_fg->output();
    MS_EXCEPTION_IF_NULL(output);
    const auto &output_abs = output->abstract();
    MS_EXCEPTION_IF_NULL(output_abs);
    auto py_forward_result = pipeline::BaseRefToPyDataWithUserData(forward_result, output_abs);
    py::tuple ret_tuple = py::cast<py::tuple>(py_forward_result);
    if (!py::isinstance<py::tuple>(ret_tuple) || !ret_tuple.size()) {
      MS_LOG(EXCEPTION) << "Forward output is not valid for fg: " << forward_fg->ToString()
                        << " , output: " << py::str(py_forward_result);
    }
    std::transform(ret_tuple.begin() + 1, ret_tuple.end(), std::back_inserter(grad_param->added_args),
                   [](const auto &element) { return PyObjToValue(py::cast<py::object>(element)); });
    return ret_tuple[kIndex0];
  }
}

bool IsValidAbstract(const AbstractBasePtr &prim_abstract) {
  if (prim_abstract == nullptr) {
    return false;
  } else if (prim_abstract->isa<abstract::AbstractRefTensor>()) {
    const auto ref_abs = prim_abstract->cast_ptr<abstract::AbstractRefTensor>();
    MS_EXCEPTION_IF_NULL(ref_abs);
    return !ref_abs->is_view() && !ref_abs->is_inplace();
  } else if (prim_abstract->isa<abstract::AbstractTensor>()) {
    return true;
  } else if (prim_abstract->isa<abstract::AbstractSequence>()) {
    const auto &elements = prim_abstract->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(),
                       [](const AbstractBasePtr &element) { return IsValidAbstract(element); });
  }
  return false;
}
}  // namespace

std::pair<bool, FuncGraphPtr> GetBpropGraph(const pynative::GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_EXCEPTION_IF_NULL(grad_param->op_grad_info);

  FuncGraphPtr after_opt_fg = nullptr;
  FuncGraphPtr forward_fg = nullptr;
  BpropGeneratorPtr jit_adgrad_processer = nullptr;

  // Determine if forward result is needed, eg: second grad for high grad no need
  const bool need_forward_result = (grad_param->op_grad_info->out_value == nullptr);
  // Determine if forward node reuse is needed, first grad for high grad is ir_grad, no need reuse forward node
  const bool need_reuse_forward_node = need_forward_result && !grad_param->is_high_order;
  grad_param->graph_cache_key = grad_param->graph_cache_key + (grad_param->is_high_order ? ".high_order" : "");
  MS_LOG(INFO) << "Get Bprop from fprop, need forward result: " << need_forward_result
               << " , need reuse forward node: " << need_reuse_forward_node
               << " , is high order: " << grad_param->is_high_order << " , cache key: " << grad_param->graph_cache_key;

  // 1. Check cache for existing graphs
  const auto it = pass_grad_graph_.find(grad_param->graph_cache_key);
  bool cache_hit = it != pass_grad_graph_.end();
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache, cache key: " << grad_param->graph_cache_key;
    std::tie(forward_fg, after_opt_fg) = it->second;
  } else {
    // Generate forward graph with reused cnode as output
    jit_adgrad_processer = std::make_shared<BpropGenerator>(
      BasicClone(grad_param->fg), grad_param->op_grad_info->input_abs, grad_param->op_grad_info->input_value,
      grad_param->op_grad_info->out_abs, need_reuse_forward_node);
    MS_LOG(INFO) << "Start generating forward graph.";
    forward_fg = jit_adgrad_processer->GenerateForwardGraph(grad_param->source_fg, grad_param->is_control_flow);
    MS_LOG(INFO) << "Forward graph generated successfully.";
    pynative::CommonUtils::DumpGraphIR("opt_forward.ir", forward_fg);
  }

  // 2. Execute forward graph if needed
  // Prepare argument list for graph execution
  VectorRef arg_list;
  std::transform(grad_param->op_grad_info->input_value.begin(), grad_param->op_grad_info->input_value.end(),
                 std::back_inserter(arg_list), [](const ValuePtr &value) { return value; });
  ValuePtr forward_output_value = grad_param->op_grad_info->out_value;
  AbstractBasePtr origin_forward_output_abs = grad_param->op_grad_info->out_abs;
  MS_EXCEPTION_IF_NULL(origin_forward_output_abs);
  MS_EXCEPTION_IF_NULL(forward_fg);
  if (need_forward_result) {
    MS_LOG(INFO) << "Start run forward graph result";
    const auto &output = forward_fg->output();
    MS_EXCEPTION_IF_NULL(output);
    const auto &output_abs = output->abstract();
    MS_EXCEPTION_IF_NULL(output_abs);
    if (need_reuse_forward_node) {
      // {prim::kPrimMakeTuple, origin_forward_output, {prim::kPrimMakeTuple, reuse_cnode1, reuse_cnode2, ...}}
      auto tuple_output_abstract = output_abs->cast<abstract::AbstractTuplePtr>();
      if (tuple_output_abstract == nullptr || tuple_output_abstract->size() == 0) {
        MS_LOG(EXCEPTION) << "Invalid output abstract: " << output_abs->ToString();
      }
      auto node_abstracts = tuple_output_abstract->elements();
      node_abstracts[kIndex0] = origin_forward_output_abs;
      forward_fg->output()->set_abstract(std::make_shared<abstract::AbstractTuple>(node_abstracts));
    } else {
      forward_fg->output()->set_abstract(origin_forward_output_abs);
    }
    auto forward_result = GetGraphResult(forward_fg, arg_list, cache_hit, grad_param->graph_cache_key);
    py::object py_forward_result =
      HandleForwardResult(forward_result, forward_fg, origin_forward_output_abs, grad_param, need_reuse_forward_node);
    MS_LOG(DEBUG) << "Run forward graph get result: " << py::str(py_forward_result);
    forward_output_value = PyObjToValue(py_forward_result);
    grad_param->op_grad_info->out_value = forward_output_value;
  }

  // 3. Update grad_param info about forward output value
  grad_param->args = arg_list;
  MS_EXCEPTION_IF_NULL(forward_output_value);
  MS_EXCEPTION_IF_NULL(grad_param->op_grad_info->out_value);
  AbstractBasePtr real_forward_output_abs = forward_output_value->ToAbstract();
  MS_EXCEPTION_IF_NULL(origin_forward_output_abs);
  if (origin_forward_output_abs->isa<abstract::AbstractAny>()) {
    grad_param->op_grad_info->out_abs = pynative::CommonUtils::SetAbstractValueToAnyValue(real_forward_output_abs);
  }
  grad_param->jit_out_has_dict = JitOutputHasDict(grad_param->op_grad_info->out_abs);

  // 4. Store forward_graph and bprop
  if (!cache_hit) {
    MS_LOG(INFO) << "Start generating brop graph.";
    jit_adgrad_processer->set_forward_output_abs(grad_param->op_grad_info->out_abs);
    after_opt_fg = jit_adgrad_processer->GenerateBpropGraph();
    MS_LOG(INFO) << "Start optimizing brop graph.";
    pynative::CommonUtils::DumpGraphIR("opt_backward_before_opt.ir", after_opt_fg);
    // Cache original bprop graph to do invalid view inplace dout check
    if (after_opt_fg->has_flag(opt::irpass::kFlagNeedCheckViewInplaceDoutBprop)) {
      auto check_invalid_dout_level = common::GetCompileConfig("CHECK_INVALID_VIEW_INPLACE_DOUT_LEVEL");
      if (check_invalid_dout_level == "" || check_invalid_dout_level == opt::irpass::kCheckDoutLevelSceneOne) {
        original_bprop_graph[grad_param->graph_cache_key] = BasicClone(after_opt_fg);
      }
      after_opt_fg->erase_flag(opt::irpass::kFlagNeedCheckViewInplaceDoutBprop);
    }
    after_opt_fg = OptimizeBpropGraph(after_opt_fg, grad_param);
    MS_LOG(INFO) << "Bprop graph generated successfully.";
    if (grad_param->is_jit_graph) {
      pass_grad_graph_[grad_param->graph_cache_key] = {forward_fg, after_opt_fg};
    }
    pynative::CommonUtils::DumpGraphIR("opt_backward.ir", after_opt_fg);
  }
  return std::make_pair(cache_hit, after_opt_fg);
}

void ClearGradCache() {
  pass_grad_graph_.clear();
  jit_forward_resource.clear();
  original_bprop_graph.clear();
  check_invalid_dout_bprop_graph.clear();
}

void CheckBpropGraphHasInvalidDout(const std::string &cache_key, const std::vector<bool> &need_grads) {
  const auto &it_for_ori_bprop_graph = original_bprop_graph.find(cache_key);
  if (it_for_ori_bprop_graph == original_bprop_graph.end()) {
    return;
  }
  // Using cache_key and need_grad_indexes as final key to get check result
  std::ostringstream oss;
  oss << cache_key;
  for (bool b : need_grads) {
    oss << (b ? '1' : '0');
  }
  std::string check_dout_key = oss.str();
  // Has checked before and passed
  if (check_invalid_dout_bprop_graph.find(check_dout_key) != check_invalid_dout_bprop_graph.end()) {
    return;
  }
  auto original_bprop = it_for_ori_bprop_graph->second;
  MS_EXCEPTION_IF_NULL(original_bprop);
  MS_LOG(INFO) << "Do invalid view inpalce dout check for cache_key: " << check_dout_key;
  mindspore::opt::irpass::CheckBpropGraphHasInvalidDoutHelper(original_bprop, need_grads);
  check_invalid_dout_bprop_graph.insert(check_dout_key);
}

void BpropGenerator::ReuseCustomBpropForwardOutput(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg) {
  const auto &forward_fg_iter = k_fg->transforms().find("custom_bprop_primal");
  if (forward_fg_iter == k_fg->transforms().end()) {
    return;
  }
  auto primal_forward_fg = forward_fg_iter->second.func_graph();
  for (auto node : TopoSort(top_fg->output())) {
    if (node == nullptr) {
      continue;
    }
    auto forward_output = node->cast<CNodePtr>();
    if (forward_output == nullptr) {
      continue;
    }
    if (GetValueNode<FuncGraphPtr>(forward_output->input(0)) != primal_forward_fg) {
      continue;
    }
    auto &forward_output_abs = forward_output->abstract();
    MS_EXCEPTION_IF_NULL(forward_output_abs);
    MS_LOG(INFO) << "Reuse custom bprop's forward output node: " << forward_output->DebugString()
                 << ", with index: " << fprop_sub_fgs_.size();
    (void)fprop_sub_fgs_.emplace_back(k_fg);
    (void)replace_nodes_.emplace_back(forward_output);
    (void)replace_nodes_abs_.emplace_back(forward_output_abs);
  }
}

void BpropGenerator::ReusePrimalCNode(const FuncGraphPtr &k_fg, const FuncGraphPtr &top_fg) {
  // Find primal cnode for this fprop
  const auto &primal_cnode_iter = k_fg->transforms().find("primal_cnode");
  if (primal_cnode_iter == k_fg->transforms().end()) {
    return;
  }
  // Filter control flow graph and unsupported prim
  const auto &primal_cnode = primal_cnode_iter->second.primal_cnode();
  MS_EXCEPTION_IF_NULL(primal_cnode);
  if (primal_cnode->func_graph() != top_fg || IsUnSupportPrim(primal_cnode)) {
    return;
  }
  // Process primal abstract
  const auto &prim_abstract = primal_cnode->abstract();
  if (!IsValidAbstract(prim_abstract)) {
    return;
  }
  MS_LOG(INFO) << "Reuse forward output node: " << primal_cnode->DebugString()
               << ", with index: " << fprop_sub_fgs_.size();
  (void)fprop_sub_fgs_.emplace_back(k_fg);
  (void)replace_nodes_.emplace_back(primal_cnode);
  (void)replace_nodes_abs_.emplace_back(prim_abstract);
}

void BpropGenerator::Init() {
  basic_graph_ = std::make_shared<FuncGraph>();
  basic_graph_->debug_info()->set_name("bprop_builder");

  // Generate bprop function: basic_graph_(inputs, dout) ==> dins
  // (result, bprop) = fprop_graph_(inputs)
  auto fprop_app_inputs = ProcessParam(basic_graph_, input_abs_, input_value_);
  fprop_app_inputs.insert(fprop_app_inputs.begin(), NewValueNode(fprop_graph_));
  // Get bprop from fprop_fg, it is 2nd output of fprop_fg
  auto fprop_app = basic_graph_->NewCNode(fprop_app_inputs);
  auto get_bprop = basic_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

  // (df, dinputs) = bprop(dout)
  // Get dinputs from calling bprop funcgraph
  AnfNodePtrList node_list{get_bprop};
  auto dout = basic_graph_->add_parameter();
  dout->set_abstract(out_abs_);
  (void)node_list.emplace_back(dout);
  auto call_bprop = basic_graph_->NewCNode(node_list);
  AnfNodePtrList actual_out{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_abs_.size(); ++i) {
    // Index 0 env, skip
    auto out =
      basic_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_bprop, NewValueNode(SizeToLong(i + 1))});
    (void)actual_out.emplace_back(out);
  }
  basic_graph_->set_output(basic_graph_->NewCNode(actual_out));
  ClearFuncGraphCNodeAbstract(basic_graph_);
  pynative::CommonUtils::DumpGraphIR("opt_before.ir", basic_graph_);

  if (!need_reuse_forward_node_) {
    return;
  }

  // Find necessary sub fprop graphs
  auto primal_fg_iter = fprop_graph_->transforms().find("primal");
  if (primal_fg_iter == fprop_graph_->transforms().end()) {
    return;
  }
  auto primal_fg = primal_fg_iter->second.func_graph();
  MS_EXCEPTION_IF_NULL(primal_fg);
  for (const auto &node : TopoSort(basic_graph_->return_node(), SuccDeeperSimple)) {
    // Check fprop graph for each prim
    auto k_fg = GetValueNode<FuncGraphPtr>(node);
    if (!k_fg) {
      continue;
    }
    if (k_fg->has_flag(opt::irpass::kFlagNeedCheckViewInplaceDoutBprop)) {
      basic_graph_->set_flag(opt::irpass::kFlagNeedCheckViewInplaceDoutBprop, true);
    }
    ReuseCustomBpropForwardOutput(k_fg, primal_fg);
    ReusePrimalCNode(k_fg, primal_fg);
  }
  MS_LOG(INFO) << "Finish init generating basic bprop func graph for " << fprop_graph_->ToString() << ", there are "
               << fprop_sub_fgs_.size() << " forward nodes could be reused.";
}

FuncGraphPtr BpropGenerator::GenerateBpropGraph() {
  if (need_reuse_forward_node_) {
    auto back_manager = Manage({basic_graph_}, false);
    size_t index = 0;
    for (const auto &k_fg : fprop_sub_fgs_) {
      auto param = basic_graph_->add_parameter();
      auto output_cnode = k_fg->output()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      auto forward_output_node = output_cnode->input(kIndex1);
      back_manager->Replace(forward_output_node, param);
      param->set_abstract(replace_nodes_abs_[index++]);
    }
  }
  return basic_graph_;
}

FuncGraphPtr BpropGenerator::GenerateForwardGraph(const FuncGraphPtr &jit_forward_graph, bool do_renormalize) {
  if (!need_reuse_forward_node_) {
    return OptimizeForwardGraph(BasicClone(jit_forward_graph), do_renormalize);
  }
  auto primal_fg_iter = fprop_graph_->transforms().find("primal");
  if (primal_fg_iter == fprop_graph_->transforms().end()) {
    return OptimizeForwardGraph(BasicClone(jit_forward_graph), do_renormalize);
  }
  // Need modify forward output
  // From {kPrimReturn, original_output} ==> {kPrimReturn, {kPrimMakeTuple, original_output, reused_cnodes}}
  const auto &primal_fg = primal_fg_iter->second.func_graph();
  MS_EXCEPTION_IF_NULL(primal_fg);
  pynative::CommonUtils::DumpGraphIR("primal_graph.ir", primal_fg);
  const auto &params = primal_fg->parameters();
  if (params.size() != input_abs_.size()) {
    MS_LOG(EXCEPTION) << "Unmatched param size for primal_fg: " << primal_fg->ToString();
  }
  for (size_t index = 0; index < input_abs_.size(); ++index) {
    auto param = params[index]->cast<ParameterPtr>();
    param->set_abstract(input_abs_[index]);
  }

  MS_LOG(INFO) << "Start appending reused nodes to forward graph output.";
  // {Primal_fg(inputs) = foward_result} ==> {Primal_fg(inputs) = (foward_result, reused nodes)}
  // Get original output node and abstract, and merge original output node and used forward nodes to return node.
  auto original_output_node = primal_fg->output();
  MS_EXCEPTION_IF_NULL(original_output_node);
  AnfNodePtrList fprop_forward_outputs{NewValueNode(prim::kPrimMakeTuple), original_output_node};
  fprop_forward_outputs.insert(fprop_forward_outputs.end(), replace_nodes_.begin(), replace_nodes_.end());
  auto merge_node = primal_fg->NewCNode(std::move(fprop_forward_outputs));
  primal_fg->set_output(merge_node);
  auto forward_fg = BasicClone(primal_fg);
  primal_fg->set_output(original_output_node);
  MS_LOG(INFO) << "Finish appending reused nodes to forward graph output.";
  return OptimizeForwardGraph(forward_fg, true);
}

void BpropGenerator::set_forward_output_abs(const abstract::AbstractBasePtr &forward_abs) {
  if (basic_graph_->parameters().empty()) {
    return;
  }
  auto input_value_size = input_value_.size();
  auto &dout_param = basic_graph_->parameters()[input_value_size];
  dout_param->set_abstract(forward_abs);
  PlantFuncGradBpropGraphDout(basic_graph_, input_value_size, forward_abs);
}
}  // namespace ad
}  // namespace mindspore
