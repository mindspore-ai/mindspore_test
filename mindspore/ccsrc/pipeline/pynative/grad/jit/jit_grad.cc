/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/grad/jit/jit_grad.h"

#include <utility>
#include "frontend/optimizer/ad/grad.h"
#include "op_def/structure_op_name.h"
#include "op_def/framework_op_name.h"
#include "op_def/sequence_ops.h"
#include "pipeline/pynative/grad/grad_utils.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/jit/ps/pipeline.h"
#include "ir/func_graph_cloner.h"
#include "frontend/expander/bprop/bprop.h"

namespace mindspore {
namespace pynative {
namespace {
constexpr char kAddedValue[] = "added_value";

const mindspore::HashSet<std::string> kExpanderWhiteList{
  kVmapStackAssignOpName,
  kVmapUnstackAssignOpName,
  kPyExecuteOpName,
  kPrintOpName,
};

py::tuple ConvertDictArgs(const py::object &args) {
  if (!py::isinstance<py::tuple>(args)) {
    MS_LOG(EXCEPTION) << "Args should be tuple but got: " << py::str(args);
  }
  auto tuple_args = py::cast<py::tuple>(args);
  py::list new_args;
  for (const auto &e : tuple_args) {
    auto element = py::cast<py::object>(e);
    if (py::isinstance<py::dict>(element)) {
      MS_LOG(INFO) << "Convert dict input " << py::str(element) << " to tuple with values.";
      new_args.append(py::reinterpret_steal<py::tuple>(PyDict_Values(element.ptr())));
    } else {
      new_args.append(element);
    }
  }
  return py::cast<py::tuple>(new_args);
}

FrontendOpRunInfoPtr GetOpRunInfo(const py::args &args, const std::string &graph_phase,
                                  const FuncGraphPtr &jit_forward_graph) {
  auto op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->requires_grad = true;
  op_run_info->is_jit_input = true;
  op_run_info->base_op_run_info.op_name = graph_phase;
  // Dict input for graph should be converted to tuple after compile.
  const auto &new_args = ConvertDictArgs(args);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, new_args);
  // Set input abs
  const auto &original_params = jit_forward_graph->parameters();
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->op_grad_info->input_abs[i] = original_params[i]->abstract();
  }
  return op_run_info;
}

size_t GetTensorNumFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    // Is a tensor
    constexpr size_t kTensorOutputNum = 1;
    return kTensorOutputNum;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    return std::accumulate(abs_seq.begin(), abs_seq.end(), 0, [](size_t out_num, const abstract::AbstractBasePtr &abs) {
      return out_num + GetTensorNumFromAbstract(abs);
    });
  }
  if (abs->isa<abstract::AbstractCSRTensor>()) {
    // Currently, CSRTensor only supports 2-D matrix (shape has 2 values). 5 outputs = 3 Tensors + 2 shape values.
    constexpr size_t kCSRTensorOutputNum = 5;
    return kCSRTensorOutputNum;
  }
  if (abs->isa<abstract::AbstractCOOTensor>()) {
    // Currently, COOTensor only supports 2-D matrix (shape has 2 values). 4 outputs = 2 Tensors + 2 shape values.
    constexpr size_t kCOOTensorOutputNum = 4;
    return kCOOTensorOutputNum;
  }
  return 0;
}

bool IsGraphDynamic(const FuncGraphPtr &func_graph) {
  for (const auto &param : func_graph->parameters()) {
    if (param->isa<Parameter>() && !param->cast<ParameterPtr>()->has_default()) {
      const auto &abs = param->abstract();
      if (abs != nullptr && abs->BuildShape()->IsDynamic()) {
        return true;
      }
    }
  }
  MS_EXCEPTION_IF_NULL(func_graph->output());
  if (auto abs = func_graph->output()->abstract(); abs != nullptr && abs->BuildShape()->IsDynamic()) {
    return true;
  }
  return false;
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
}  // namespace

void Jit::GetInputArgsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                           AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_nodes);
  MS_EXCEPTION_IF_NULL(grad_executor);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    const auto &input_i_value = op_run_info->op_grad_info->input_value[i];
    const auto &id = PyNativeAlgo::Common::GetIdByValue(input_i_value);
    const auto &input_i_node = grad_executor->GetInput(input_i_value, id);
    MS_EXCEPTION_IF_NULL(input_i_node);
    MS_LOG(DEBUG) << "The input " << i << " id " << id << " , node is: " << input_i_node->DebugString();
    (void)input_nodes->emplace_back(input_i_node);
  }
}

void Jit::GetWeightsNode(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                         const FuncGraphPtr &jit_forward_graph, AnfNodePtrList *input_nodes) const {
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(input_nodes);
  const auto &top_cell = grad_executor->top_cell();
  const auto &graph_info = top_cell->graph_info_map().at(top_cell->fg());
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get weights info of jit
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  const auto &original_params = jit_forward_graph->parameters();
  size_t params_size = original_params.size();
  MS_EXCEPTION_IF_NULL(op_run_info);
  for (size_t i = 0; i < params_size; ++i) {
    if (i < op_run_info->input_size) {  // non-weights node.
      continue;
    }
    // Must weight param
    auto param = original_params[i]->cast<ParameterPtr>();
    const auto tensor_value = PyNativeAlgo::Common::GetTensorFromParam(original_params[i]);
    MS_EXCEPTION_IF_NULL(tensor_value);
    const auto it = graph_info->weight_params.find(tensor_value->id());
    if (it != graph_info->weight_params.end()) {
      param = it->second;
    } else {
      top_cell->fg()->add_parameter(param);
      if (param->debug_info() != nullptr) {
        param->debug_info()->set_name(param->name());
      }
      top_cell->SetParamNodeMapInGraphInfoMap(tensor_value->id(), param, true);
    }
    (void)input_nodes->emplace_back(param);
    MS_LOG(DEBUG) << "Top graph set free parameter " << param->DebugString() << ". Its default value is "
                  << tensor_value->ToString() << ". Its name is: " << param->name();
  }
}

void Jit::MakeCNodeForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                          const FuncGraphPtr &jit_forward_graph, CNodePtr *jit_cnode) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  // Get input node info of jit
  AnfNodePtrList input_nodes{NewValueNode(jit_forward_graph)};
  MS_EXCEPTION_IF_NULL(grad_executor);
  GetInputArgsNode(op_run_info, grad_executor, &input_nodes);
  // Get weights node info of jit.
  GetWeightsNode(op_run_info, grad_executor, jit_forward_graph, &input_nodes);
  // Make a CNode which includes jit fprop graph and inputs node
  MS_EXCEPTION_IF_NULL(jit_cnode);
  MS_EXCEPTION_IF_NULL(grad_executor->top_cell()->fg());
  *jit_cnode = grad_executor->top_cell()->fg()->NewCNode(input_nodes);
  MS_EXCEPTION_IF_NULL(jit_forward_graph->output());
  (*jit_cnode)->set_abstract(jit_forward_graph->output()->abstract());
  MS_LOG(DEBUG) << "Make jit forward CNode: " << (*jit_cnode)->DebugString();
}

GradParamPtr Jit::CreateJitGradParam(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                     const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  PyNativeAlgo::Common::SetGraphInputAndWeightsInfo(op_run_info, jit_forward_graph);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  op_run_info->op_grad_info->out_abs = jit_forward_graph->output()->abstract();

  auto grad_param = std::make_shared<GradParam>(op_run_info->op_grad_info, grad_executor->use_dynamic_shape_process());
  grad_param->is_jit_graph = true;
  grad_param->is_control_flow = compile_info_.is_control_flow_;
  // As long as the jit is in the process of dynamic shape,
  // let it run actor execution to avoid backend pass
  grad_param->is_jit_self_dynamic_shape = compile_info_.is_dynamic_shape_;

  grad_param->fg = jit_grad_graph;
  grad_param->source_fg = jit_forward_graph;
  grad_param->graph_cache_key = graph_phase_;
  grad_param->jit_out_has_dict = JitOutputHasDict(op_run_info->op_grad_info->out_abs);
  return grad_param;
}

void Jit::RecordForwardGraphForJit(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                                   const FuncGraphPtr &jit_forward_graph) const {
  int save_graphs = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    CNodePtr jit_cnode = nullptr;
    MakeCNodeForJit(op_run_info, grad_executor, jit_forward_graph, &jit_cnode);
    MS_EXCEPTION_IF_NULL(jit_cnode);
    const auto &out_id = PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
    const auto &top_cell = grad_executor->top_cell();
    top_cell->SetNodeMapInGraphInfoMap(out_id, jit_cnode);
  }
}

void Jit::GradJitInner(const FrontendOpRunInfoPtr &op_run_info, const GradExecutor *grad_executor,
                       const FuncGraphPtr &jit_forward_graph, const FuncGraphPtr &jit_grad_graph) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(grad_executor);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  MS_EXCEPTION_IF_NULL(jit_forward_graph->output());
  const auto &forward_output_abs = jit_forward_graph->output()->abstract();
  MS_EXCEPTION_IF_NULL(forward_output_abs);

  // Step 1: Get jit op info
  const auto &top_cell = grad_executor->top_cell();
  MS_EXCEPTION_IF_NULL(top_cell);
  top_cell->GetOpInfo(op_run_info->op_grad_info, op_run_info->base_op_run_info.op_name, true);

  auto &&grad_param = CreateJitGradParam(op_run_info, grad_executor, jit_forward_graph, jit_grad_graph);
  auto auto_grad_cell_ptr = grad_executor->top_cell()->auto_grad_cell_ptr();
  MS_EXCEPTION_IF_NULL(auto_grad_cell_ptr);
  const auto &dynamic_shape = grad_executor->dynamic_shape();
  MS_EXCEPTION_IF_NULL(dynamic_shape);
  dynamic_shape->CheckNodeDynamic(grad_executor->top_cell(), grad_param->op_grad_info, grad_param->graph_cache_key);
  if (!auto_grad_cell_ptr->KPynativeWithFProp(grad_param)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for jit cnode";
  }
  op_run_info->real_out = grad_param->op_grad_info->out_value;
  // Step 2: Flatten output value
  ValuePtrList flatten_actual_output_list;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(op_run_info->real_out, false, true, &flatten_actual_output_list);
  auto flatten_replace_value = std::make_shared<ValueTuple>(flatten_actual_output_list);
  auto pre_top_cell = PyNativeAlgo::Common::FindPreTopcell(grad_executor, op_run_info->op_grad_info,
                                                           op_run_info->op_grad_info->op_info, flatten_replace_value);
  // Update output related info
  op_run_info->op_grad_info->out_value = flatten_replace_value;
  MS_LOG(DEBUG) << "jit actual output value: " << op_run_info->real_out->ToString() << ", output id "
                << PyNativeAlgo::Common::GetIdByValue(op_run_info->real_out);
  PyNativeAlgo::Common::UpdateGradOpInfo(grad_executor, grad_param->op_grad_info, pre_top_cell, true);
  RecordForwardGraphForJit(op_run_info, grad_executor, jit_forward_graph);
  if (dynamic_shape->enable_unknown_shape() && forward_output_abs->BuildShape()->IsDynamic()) {
    MS_LOG(DEBUG) << "Set jit unknown shape out to abs cache";
    dynamic_shape->SaveUnknownShapeAbsFromJit(op_run_info->real_out, forward_output_abs, 0);
  }
  top_cell->set_need_do_final_opt(true);
  top_cell->set_has_call_graph(grad_executor->use_dynamic_shape_process());
  top_cell->set_has_control_flow(compile_info_.is_control_flow_);
  top_cell->set_jit_out_has_dict(grad_param->jit_out_has_dict);
}

bool Jit::GetJitGradGraph(const pipeline::ResourcePtr &resource) {
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  graph_phase_ = graph_executor->phase();
  MS_LOG(DEBUG) << "The phase of current pipeline graph is: " << graph_phase_;
  // Exporting graph in PyNative mode or only running forward process no need to do this action.
  const auto &pynative_grad_executor = PyNativeExecutor::grad_executor();
  if (graph_phase_.find("export") == 0 || !pynative_grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "When exporting graph or only running forward process";
    return true;
  }

  // Set Primal graph
  MS_EXCEPTION_IF_NULL(resource);
  auto jit_forward_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  auto primal_fg = BasicClone(jit_forward_graph);
  graph_executor->SetJitPrimalFuncGraph(primal_fg, graph_phase_);
  // Using adgrad to generate fprop func graph for jit function in pynative mode
  bool is_control_flow = PyNativeAlgo::Common::IsControlFlowGraph(jit_forward_graph);
  auto grad_graph =
    ad::Grad(is_control_flow ? primal_fg : jit_forward_graph, opt::Optimizer::MakeEmptyOptimizer(resource));
  MS_EXCEPTION_IF_NULL(grad_graph);
  graph_executor->SetJitGradGraph(grad_graph, graph_phase_);
  // Set jit compile info
  jit_compile_info_[graph_phase_] = JitCompileInfo();
  jit_compile_info_[graph_phase_].is_control_flow_ = is_control_flow;
  jit_compile_info_[graph_phase_].is_dynamic_shape_ = IsGraphDynamic(jit_forward_graph);
  MS_LOG(DEBUG) << "Func graph is control flow: " << jit_compile_info_[graph_phase_].is_control_flow_
                << " , is dynamic shape: " << jit_compile_info_[graph_phase_].is_dynamic_shape_;
  PyNativeAlgo::Common::DumpGraphIR("jit_modify_before_forward_graph.ir", jit_forward_graph);

  // Keep roots for only keeping forward func graph in resource.
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->KeepRoots({jit_forward_graph});
  return true;
}

void Jit::Reset() { graph_phase_.clear(); }

void Jit::Clear() {
  for (auto &t : graph_phase_with_replace_info_) {
    t.second.clear();
  }
}

py::object Jit::GradJit(const py::args &args) {
  if (graph_phase_.empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain jit func graph.";
  }
  MS_LOG(DEBUG) << "jit func graph phase: " << graph_phase_;
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  const auto &grad_executor = mindspore::pynative::PyNativeExecutor::grad_executor();
  // For no grad function, execute forward graph directly
  if (!grad_executor->RequiresGrad()) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    py::object ret = executor->Run(args, py::str(graph_phase_));
    graph_phase_.clear();
    return ret;
  }
  if (auto iter = jit_compile_info_.find(graph_phase_); iter != jit_compile_info_.end()) {
    compile_info_ = iter->second;
  } else {
    MS_LOG(EXCEPTION) << "Can not find graph_phase_: " << graph_phase_ << " in jit_compile_info_. "
                      << "Please check if the corresponding forward compilation process is missing.";
  }
  // Get compiled forward graph and generate op_run_info
  FuncGraphPtr jit_forward_graph = executor->GetFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_forward_graph);
  const auto &op_run_info = GetOpRunInfo(args, graph_phase_, jit_forward_graph);
  PyNativeAlgo::Common::DumpGraphIR("jit_forward_graph.ir", jit_forward_graph);
  // Get grad graph after adgrad
  auto jit_grad_graph = executor->GetJitGradGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_grad_graph);
  // Get primal graph
  auto jit_primal_graph = executor->GetJitPrimalFuncGraph(graph_phase_);
  MS_EXCEPTION_IF_NULL(jit_primal_graph);
  if (compile_info_.is_dynamic_shape_) {
    grad_executor->set_use_dynamic_shape_process(true);
  }
  MS_LOG(DEBUG) << "Start gradjit using forward primal graph and adgrad graph";
  GradJitInner(op_run_info, grad_executor.get(), jit_primal_graph, jit_grad_graph);
  Reset();
  MS_EXCEPTION_IF_NULL(op_run_info->real_out);
  return PyNativeAlgo::DataConvert::ValueToPyObj(op_run_info->real_out);
}
}  // namespace pynative
}  // namespace mindspore
