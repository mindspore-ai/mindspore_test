/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_v.h"
namespace mindspore {
namespace ad {
mindspore::HashMap<std::string, std::pair<FuncGraphPtr, FuncGraphPtr>> pass_grad_graph_param_;
mindspore::HashMap<std::string, FuncGraphPtr> pass_grad_graph_valuenode_;
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_forward_resource;

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

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.find(kAttrRecompute) == 0;
}

bool HasRecomputedScope(const CNodePtr &node) {
  // Exclude nodes without recompute scope
  if (!WithRecomputedScope(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto recompute_attr = cnode->GetAttr(kAttrRecompute);
  return recompute_attr != nullptr && recompute_attr->isa<BoolImm>() && GetValue<bool>(recompute_attr);
}
}  // namespace

std::pair<bool, FuncGraphPtr> GetBpropGraphWithParamalization(const pynative::GradParamPtr &grad_param) {
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
  const auto it = pass_grad_graph_param_.find(grad_param->graph_cache_key);
  bool cache_hit = it != pass_grad_graph_param_.end();
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache, cache key: " << grad_param->graph_cache_key;
    std::tie(forward_fg, after_opt_fg) = it->second;
  } else {
    // Generate backward graph and forward graph with reused cnode as output
    jit_adgrad_processer = std::make_shared<BpropGenerator>(
      BasicClone(grad_param->fg), grad_param->op_grad_info->input_abs, grad_param->op_grad_info->input_value,
      grad_param->op_grad_info->out_abs, need_reuse_forward_node);

    // Generating backward_graph
    MS_LOG(INFO) << "Start generating brop graph.";
    after_opt_fg = jit_adgrad_processer->GenerateBpropGraph();
    MS_LOG(INFO) << "Start optimizing brop graph.";
    pynative::CommonUtils::DumpGraphIR("opt_backward_before_opt.ir", after_opt_fg);
    after_opt_fg = OptimizeBpropGraph(after_opt_fg, grad_param);
    pynative::CommonUtils::DumpGraphIR("opt_backward_after_opt.ir", after_opt_fg);
    jit_adgrad_processer->EreaseUnusedReuseCNode(after_opt_fg);
    MS_LOG(INFO) << "Bprop graph generated successfully.";

    // Generating forward_graph
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
      output->set_abstract(std::make_shared<abstract::AbstractTuple>(node_abstracts));
    } else {
      output->set_abstract(origin_forward_output_abs);
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
  if (origin_forward_output_abs->isa<abstract::AbstractAny>()) {
    grad_param->op_grad_info->out_abs = pynative::CommonUtils::SetAbstractValueToAnyValue(real_forward_output_abs);
  }
  grad_param->jit_out_has_dict = JitOutputHasDict(grad_param->op_grad_info->out_abs);

  // 4. Store forward_graph and bprop
  if (!cache_hit) {
    jit_adgrad_processer->SetForwardOutputAbs(grad_param->op_grad_info->out_abs, after_opt_fg);
    pynative::CommonUtils::DumpGraphIR("opt_backward.ir", after_opt_fg);
    if (grad_param->is_jit_graph) {
      pass_grad_graph_param_[grad_param->graph_cache_key] = {forward_fg, after_opt_fg};
    }
  }
  return std::make_pair(cache_hit, after_opt_fg);
}

std::pair<bool, FuncGraphPtr> GetBpropGraphWithValueNodeReplacement(const pynative::GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  FuncGraphPtr after_opt_fg = nullptr;
  // Find ad graph in cache
  const auto it = pass_grad_graph_valuenode_.find(grad_param->graph_cache_key);
  bool cache_hit = (it != pass_grad_graph_valuenode_.end());
  if (cache_hit) {
    MS_LOG(DEBUG) << "Get ad grad graph by cache";
    after_opt_fg = grad_param->is_control_flow ? it->second : BasicClone(it->second);
  } else {
    auto bprop_builder = std::make_shared<FuncGraph>();
    bprop_builder->debug_info()->set_name("bprop_builder");

    // grad_param->fg --> K(func)
    auto fprop_app_inputs =
      ProcessParam(bprop_builder, grad_param->op_grad_info->input_abs, grad_param->op_grad_info->input_value);
    fprop_app_inputs.insert(fprop_app_inputs.begin(), NewValueNode(BasicClone(grad_param->fg)));
    // (result, bprop) = K(func)(inputs)
    auto fprop_app = bprop_builder->NewCNode(fprop_app_inputs);
    // Get bprop from fprop_fg, it is 2th output of fprop_fg
    auto get_bprop = bprop_builder->NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), fprop_app, NewValueNode(static_cast<int64_t>(kIndex1))});

    AnfNodePtrList node_list{get_bprop};
    auto dout = bprop_builder->add_parameter();
    dout->set_abstract(grad_param->op_grad_info->out_abs);
    (void)node_list.emplace_back(dout);
    // df, dinputs = bprop(dout)
    auto call_bprop = bprop_builder->NewCNode(node_list);

    AnfNodePtrList actual_out{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < grad_param->input_size; ++i) {
      // Index 0 env, skip
      auto out =
        bprop_builder->NewCNode({NewValueNode(prim::kPrimTupleGetItem), call_bprop, NewValueNode(SizeToLong(i + 1))});
      (void)actual_out.emplace_back(out);
    }
    bprop_builder->set_output(bprop_builder->NewCNode(actual_out));
    // Call pass for optimize graph, such as inline
    ClearFuncGraphCNodeAbstract(bprop_builder);
    after_opt_fg = OptimizeBpropGraph(bprop_builder, grad_param);
    PlantFuncGradBpropGraphDout(after_opt_fg, grad_param->input_size, grad_param->op_grad_info->out_abs);
    if (grad_param->is_jit_graph) {
      // Control flow no need do valuenode replacement, just return the original funcgraph
      pass_grad_graph_valuenode_[grad_param->graph_cache_key] =
        grad_param->is_control_flow ? after_opt_fg : BasicClone(after_opt_fg);
    }
    pynative::CommonUtils::DumpGraphIR("opt_backward.ir", after_opt_fg);
  }
  VectorRef arg_list;
  std::transform(grad_param->op_grad_info->input_value.begin(), grad_param->op_grad_info->input_value.end(),
                 std::back_inserter(arg_list), [](const ValuePtr &value) { return value; });
  grad_param->args = arg_list;
  return std::make_pair(cache_hit, after_opt_fg);
}

// Entrance for gradjit get bprop graph
std::pair<bool, FuncGraphPtr> GetBpropGraph(const pynative::GradParamPtr &grad_param) {
  static bool enable_valuenode_replace = (common::GetCompileConfig("PYNATIVE_JIT_GRAD_MODE") == "1");
  MS_LOG(INFO) << "Process bprop graph with enable valuenode replacement method : " << enable_valuenode_replace;
  if (enable_valuenode_replace) {
    return GetBpropGraphWithValueNodeReplacement(grad_param);
  } else {
    return GetBpropGraphWithParamalization(grad_param);
  }
}

void ClearGradCache() {
  pass_grad_graph_valuenode_.clear();
  pass_grad_graph_param_.clear();
  jit_forward_resource.clear();
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

  // Check whether top cell do recompute
  bool top_cell_do_recompute = primal_fg->has_flag(kTopCellWithRecompute);

  for (const auto &node : TopoSort(basic_graph_->return_node(), SuccDeeperSimple)) {
    // Check fprop graph for each prim
    auto k_fg = GetValueNode<FuncGraphPtr>(node);
    if (!k_fg) {
      continue;
    }
    // Find primal cnode for this fprop
    const auto &primal_cnode_iter = k_fg->transforms().find("primal_cnode");
    if (primal_cnode_iter == k_fg->transforms().end()) {
      continue;
    }
    // Filter control flow graph and unsupported prim
    const auto &primal_cnode = primal_cnode_iter->second.primal_cnode();
    MS_EXCEPTION_IF_NULL(primal_cnode);
    if (primal_cnode->func_graph() != primal_fg || IsUnSupportPrim(primal_cnode)) {
      continue;
    }
    // Do not reuse recompute sub cell's cnode
    if (!top_cell_do_recompute && HasRecomputedScope(primal_cnode)) {
      MS_LOG(DEBUG) << "Need recompute cnode: " << primal_cnode->DebugString();
      continue;
    }
    // Process primal abstract
    const auto &prim_abstract = primal_cnode->abstract();
    if (!prim_abstract || !prim_abstract->isa<abstract::AbstractTensor>()) {
      continue;
    }
    MS_LOG(DEBUG) << "Reuse forward output node: " << primal_cnode->DebugString()
                  << ", with index: " << fprop_sub_fgs_.size();
    (void)fprop_sub_fgs_.emplace_back(k_fg);
    (void)replace_nodes_.emplace_back(primal_cnode);
    (void)replace_nodes_abs_.emplace_back(prim_abstract);
  }
  MS_LOG(INFO) << "Finish init generating basic bprop func graph for " << fprop_graph_->ToString() << ", there are "
               << fprop_sub_fgs_.size() << " forward nodes could be reused.";
}

FuncGraphPtr BpropGenerator::GenerateBpropGraph() {
  if (need_reuse_forward_node_) {
    bprop_origin_param_size_ = basic_graph_->parameters().size();
    auto back_manager = Manage({basic_graph_}, false);
    size_t index = 0;
    for (const auto &k_fg : fprop_sub_fgs_) {
      auto param = basic_graph_->add_parameter();
      auto forward_output_node = k_fg->output()->cast<CNodePtr>()->input(kIndex1);
      back_manager->Replace(forward_output_node, param);
      param->set_abstract(replace_nodes_abs_[index++]);
    }
  }
  return basic_graph_;
}

// Erase unused forward_reused params after bprop_sub_fg expanded
void BpropGenerator::EreaseUnusedReuseCNode(const FuncGraphPtr &bprop_fg) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  auto manager = Manage({bprop_fg}, false);
  auto params = bprop_fg->parameters();
  auto node_users = manager->node_users();
  AnfNodePtrList new_params;
  for (size_t index = 0; index < params.size(); ++index) {
    auto param = params[index];
    // Add original forward inputs and dout
    if (index < bprop_origin_param_size_) {
      (void)new_params.emplace_back(param);
      continue;
    }
    // Add params have actual users in bprop fg
    auto use_node_size = node_users[param].size();
    if (use_node_size != 0) {
      (void)new_params.emplace_back(param);
    } else {
      MS_LOG(DEBUG) << "Unused primal cnode in bprop graph: "
                    << replace_nodes_[index - bprop_origin_param_size_]->DebugString();
      size_t origin_reuse_index = index - bprop_origin_param_size_;
      replace_nodes_[origin_reuse_index] = nullptr;
      replace_nodes_abs_[origin_reuse_index] = nullptr;
      fprop_sub_fgs_[origin_reuse_index] = nullptr;
    }
  }
  bprop_fg->set_parameters(new_params);
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
  (void)std::copy_if(replace_nodes_.begin(), replace_nodes_.end(), std::back_inserter(fprop_forward_outputs),
                     [](const AnfNodePtr &node) { return node != nullptr; });
  auto merge_node = primal_fg->NewCNode(std::move(fprop_forward_outputs));
  primal_fg->set_output(merge_node);
  auto forward_fg = BasicClone(primal_fg);
  primal_fg->set_output(original_output_node);
  MS_LOG(INFO) << "Finish appending reused nodes to forward graph output.";
  return OptimizeForwardGraph(forward_fg, true);
}

void BpropGenerator::SetForwardOutputAbs(const abstract::AbstractBasePtr &forward_abs,
                                         const FuncGraphPtr &bprop_graph) {
  if (bprop_graph->parameters().empty()) {
    return;
  }
  auto input_value_size = input_value_.size();
  auto &dout_param = bprop_graph->parameters()[input_value_size];
  dout_param->set_abstract(forward_abs);
  PlantFuncGradBpropGraphDout(bprop_graph, input_value_size, forward_abs);
}
}  // namespace ad
}  // namespace mindspore
