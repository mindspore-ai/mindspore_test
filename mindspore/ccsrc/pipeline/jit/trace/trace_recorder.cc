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

#include "mindspore/ccsrc/pipeline/jit/trace/trace_recorder.h"

#include <algorithm>
#include <mutex>
#include <utility>
#include <vector>

#include "utils/ms_context.h"
#include "frontend/operator/composite/do_signature.h"
#include "include/common/debug/anf_ir_dump.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "pipeline/jit/ps/pipeline_jit.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "frontend/operator/ops_front_infer_function.h"

namespace py = pybind11;
namespace mindspore {
namespace trace {
namespace {
abstract::AbstractBasePtr GetAbstract(const py::object &obj) {
  ValuePtr val = nullptr;
  parse::ConvertData(obj, &val);
  MS_EXCEPTION_IF_NULL(val);
  const auto &abs = abstract::ToAbstract(val, nullptr, nullptr);
  MS_EXCEPTION_IF_NULL(abs);
  return abs;
}

std::string GetPyObjId(const py::object &obj) {
  py::object py_obj_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(py_obj_str)) {
    MS_LOG(INTERNAL_EXCEPTION) << "The object has no id(), [" << py::str(obj.get_type()) << "] " << py::str(obj);
  }
  return py_obj_str.cast<std::string>();
}

bool IsMutable(const py::object &obj) {
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

CNodePtr GenerateCNode(const FuncGraphPtr &func_graph, const PrimitivePtr &prim, const AnfNodePtrList &args_inputs) {
  auto node_inputs = args_inputs;
  if (ops::IsPrimitiveFunction(prim->name())) {
    const auto &new_prim = std::make_shared<prim::DoTransPrimitiveFunction>(std::make_shared<Primitive>(prim->name()));
    (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(new_prim));
  } else {
    (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(prim));
  }
  return func_graph->NewCNodeInOrder(node_inputs);
}

void SyncTensor(const py::object &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    const auto &tensor = py::cast<tensor::TensorPtr>(obj);
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->data_sync();
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &obj_tuple = py::cast<py::tuple>(obj);
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      SyncTensor(obj_tuple[i]);
    }
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &obj_list = py::cast<py::list>(obj);
    for (size_t i = 0; i < obj_list.size(); ++i) {
      SyncTensor(obj_list[i]);
    }
  }
}

DebugInfoPtr GenerateDebugInfos(const py::list &file_names, const py::list &linenos, const std::string &name = "") {
  if (file_names.size() == 0 || linenos.size() == 0 || file_names.size() != linenos.size()) {
    MS_LOG(EXCEPTION) << "Wrong line info list size, " << file_names.size() << ", " << linenos.size();
  }
  DebugInfoPtr debug_info = nullptr;
  TraceInfoPtr trace_info = nullptr;
  for (size_t i = file_names.size(); i > 0; --i) {
    const auto &file_name = file_names[i - 1];
    const auto &lineno = linenos[i - 1];
    const auto location = std::make_shared<Location>(py::cast<std::string>(file_name), py::cast<int>(lineno), 0,
                                                     py::cast<int>(lineno), 0, "", std::vector<std::string>());
    debug_info = std::make_shared<DebugInfo>(location);
    if (trace_info != nullptr) {
      debug_info->set_trace_info(trace_info);
    } else if (!name.empty()) {  // Set name for root debug info.
      debug_info->set_name(name);
    }
    trace_info = MakeTraceInfo<TraceOpt>(debug_info);
  }
  return debug_info;
}
}  // namespace

void Capture(const py::args &args, py::object *res) {
  if (!IsTracing()) {
    return;
  }
  *res = CaptureRun(py::args(py::tuple(args[1])), *res, args[0]);
}

void Capture(const py::list &args, py::object *res, std::string class_name) {
  if (!IsTracing()) {
    return;
  }
  const py::object &prim_py = python_adapter::CallPyFn("mindspore.ops", class_name);
  *res = CaptureRun(py::args(py::tuple(args)), *res, prim_py);
}

py::object CaptureRun(const py::args &args, const py::object &res, const py::object &prim_py) {
  // Capture node from trace func.
  auto jit_context = python_adapter::CallPyFn("mindspore.common.jit_context", "jit_context");
  std::string method = "run_op";
  return jit_context.attr(method.c_str())(prim_py, res, *args);
}

bool IsTracing() { return trace::TraceRecorder::GetInstance()->BuildingTraceGraph(); }

void TraceRecorder::Clear() {
  // Clear the AnfNode in python object.
  py_obj_node_map_.clear();
  std::stack<FuncGraphPtr>().swap(graph_stack_);
  side_effect_nodes_.clear();
  args_ = py::tuple();
  phase_.clear();
}

FuncGraphPtr TraceRecorder::InitTopGraph(const DebugInfoPtr &debug_info) {
  if (!graph_stack_.empty()) {
    MS_LOG(EXCEPTION) << "A trace graph is already created, Please check if there are nested trace functions";
  }
  auto fg_debug_info = std::make_shared<GraphDebugInfo>(MakeTraceInfo<TraceOpt>(debug_info));
  const auto new_graph = std::make_shared<FuncGraph>(std::move(fg_debug_info));
  graph_stack_.push(new_graph);
  return new_graph;
}

void TraceRecorder::BeginGraph(const py::object &func_name, const py::object &phase, const py::list &file_names,
                               const py::list &linenos, const py::args &args) {
  phase_ = py::cast<std::string>(phase);
  args_ = args;
  // Normalize the name and set as debug name.
  auto function_name = py::cast<std::string>(func_name);
  std::replace(function_name.begin(), function_name.end(), '.', '_');
  function_name += "__trace_";
  const auto debug_info = GenerateDebugInfos(file_names, linenos, function_name);
  const auto new_graph = InitTopGraph(debug_info);
  MS_LOG(DEBUG) << "Start build graph, " << new_graph << "/" << new_graph->ToString() << ", arg size: " << args.size()
                << ", args: " << py::str(py::cast<py::object>(args)) << ", phase_: " << phase_;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &param = new_graph->add_parameter();
    std::stringstream param_name_buffer;
    param_name_buffer << "arg" << i;
    const auto &param_name = param_name_buffer.str();
    param->set_name(param_name);
    if (param->debug_info() != nullptr) {
      param->debug_info()->set_name(param_name);
    }
    SetNode(args[i], param, debug_info);
  }
}

FuncGraphPtr TraceRecorder::BuildEndGraph(const py::list &file_names, const py::list &linenos,
                                          const py::args &output_args, bool nested) {
  const auto &func_graph = graph_stack_.top();
  MS_LOG(DEBUG) << "End build graph, " << func_graph << "/" << func_graph->ToString()
                << ", output_args: " << py::str(py::cast<py::object>(output_args)) << ", phase_: " << phase_;
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  if (output_args.size() == 1) {  // Maybe function output.
    const auto &fn_res = output_args[0];
    const auto &output_node = GetNode(fn_res, debug_info);
    func_graph->set_output(output_node);
    parse::AttachIsolatedNodes(func_graph, side_effect_nodes_);
  } else {  // Definitely jit block multiple outputs.
    AnfNodePtrList make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < output_args.size(); ++i) {
      (void)make_tuple_inputs.emplace_back(GetNode(output_args[i], debug_info));
    }
    auto make_tuple_node = func_graph->NewCNode(make_tuple_inputs);
    func_graph->set_output(make_tuple_node);
    parse::AttachIsolatedNodes(func_graph, side_effect_nodes_);
  }
  MS_EXCEPTION_IF_NULL(func_graph->return_node());
  if (func_graph->return_node()->debug_info() != nullptr) {
    func_graph->return_node()->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "End build graph, " << func_graph << "/" << func_graph->ToString() << ", phase_: " << phase_;
#ifdef ENABLE_DUMP_IR
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("jit_trace_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  if (nested) {
    graph_stack_.pop();
    Clear();
  }
  return func_graph;
}

void TraceRecorder::EndGraph(const py::list &file_names, const py::list &linenos, const py::args &output_args) {
  const auto &func_graph = BuildEndGraph(file_names, linenos, output_args);
  // Run compile pipeline with func graph.
  auto graph_executor = pipeline::GetExecutor();
  (void)graph_executor->CompileInner(func_graph, args_, py::dict(), phase_, true);
  MS_LOG(DEBUG) << "End compile pipeline.";
  graph_stack_.pop();
  Clear();
}

py::object TraceRecorder::RunGraph(const py::object &phase, const py::tuple &args) {
  MS_LOG(DEBUG) << "Run graph, arg size: " << args.size() << ", args: " << py::str(py::cast<py::object>(args))
                << ", phase: " << phase;
  auto graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  py::object res = graph_executor->Run(args, phase);
  if (IS_OUTPUT_ON(mindspore::kDebug)) {
    SyncTensor(res);
    MS_LOG(DEBUG) << "forward res: " << py::str(res);
  }
  int mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto executor = pynative::PyNativeExecutor::GetInstance();
  if (mode == kPynativeMode && executor->RequiresGrad()) {
    executor->grad_executor()->jit()->set_graph_phase(py::cast<std::string>(phase));
    FuncGraphPtr jit_fg = graph_executor->GetFuncGraph(py::cast<std::string>(phase));
    MS_EXCEPTION_IF_NULL(jit_fg);
    if (args.size() > jit_fg->parameters().size()) {
      MS_LOG(EXCEPTION) << "The number of inputs: " << args.size()
                        << " should not greater than the number of parameters,which is : "
                        << jit_fg->parameters().size()
                        << ". Please make sure all of the inputs were used in trace block.";
    }
    executor->GradJit(args);
    // Update forward graph with fprop graph.
    FuncGraphPtr grad_jit_fg = graph_executor->GetJitGradGraph(py::cast<std::string>(phase));
    MS_EXCEPTION_IF_NULL(grad_jit_fg);
#ifdef ENABLE_DUMP_IR
    const auto &context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("jit_trace_run_" + jit_fg->ToString() + ".ir", jit_fg);
      DumpIR("jit_trace_run_grad_" + grad_jit_fg->ToString() + ".ir", grad_jit_fg);
    }
#endif
    MS_LOG(DEBUG) << "jit_fg: " << jit_fg->ToString() << ", modify_output: " << jit_fg->modify_output()
                  << ", grad_jit_fg: " << grad_jit_fg->ToString();
    if (jit_fg->modify_output()) {
      res = py::cast<py::tuple>(res)[0];
    }
  }
  if (IS_OUTPUT_ON(mindspore::kDebug)) {
    SyncTensor(res);
    MS_LOG(DEBUG) << "return res: " << py::str(res);
  }
  return res;
}

AnfNodePtr TraceRecorder::ConvertParameterObj(const py::object &input_obj) {
  auto top_func_graph = graph_stack_.top();
  // Parameter object should not be none
  if (py::isinstance<py::none>(input_obj)) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error because obj is null.";
  }
  if (!py::hasattr(input_obj, "name")) {
    MS_LOG(EXCEPTION) << "Resolve class Parameter error: cannot find name attr for obj";
  }
  // Get the parameter name from parameter object
  auto name_attr = python_adapter::GetPyObjAttr(input_obj, "name");
  if (py::isinstance<py::none>(name_attr)) {
    MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
  }
  const auto &param_name = py::cast<std::string>(name_attr);
  auto value = parse::GetParameterValue(input_obj);
  MS_LOG(DEBUG) << "Created a new weight parameter for " << top_func_graph->ToString() << ", param: " << param_name;
  return top_func_graph->AddFvParameter(param_name, value);
}
void TraceRecorder::NewFuncGraphNode(const py::object &phase, const py::object &prim_res, const py::list &file_names,
                                     const py::list &linenos, const py::args &inputs) {
  auto graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  FuncGraphPtr jit_fg = graph_executor->GetFuncGraph(py::cast<std::string>(phase));
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  AnfNodePtrList node_inputs;
  AbstractBasePtrList abs_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    AnfNodePtr node;
    auto input_obj = inputs[i];
    bool is_parameter = py::hasattr(input_obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(input_obj);
    // When the input of a cnode is a weight, add it to the top graph.
    if (is_parameter) {
      node = ConvertParameterObj(input_obj);
    } else {
      node = GetNode(inputs[i], debug_info);
      MS_EXCEPTION_IF_NULL(node);
    }
    (void)node_inputs.emplace_back(node);
    if (node->abstract() != nullptr) {
      (void)abs_inputs.emplace_back(node->abstract());
    } else {
      (void)abs_inputs.emplace_back(GetAbstract(input_obj));
    }
    MS_LOG(DEBUG) << "Add input, " << node->DebugString();
  }
  AnfNodePtr cnode;
  (void)node_inputs.insert(node_inputs.cbegin(), NewValueNode(jit_fg));
  cnode = graph_stack_.top()->NewCNodeInOrder(node_inputs);
  if (cnode->debug_info() != nullptr) {
    cnode->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "New cnode: " << cnode->DebugString();
  SetNode(prim_res, cnode, debug_info);
}
void TraceRecorder::NewNode(const py::object &prim_obj, const py::object &prim_res, const py::list &file_names,
                            const py::list &linenos, const py::object &do_signature, const py::args &inputs) {
  MS_LOG(DEBUG) << "NewNode, prim_obj: " << py::str(prim_obj) << ", prim_res: [" << py::str(prim_res.get_type()) << "] "
                << GetPyObjId(prim_res) << "/" << py::str(prim_res) << ", inputs size: " << inputs.size()
                << ", inputs: " << py::str(py::cast<py::object>(inputs));
  const auto debug_info = GenerateDebugInfos(file_names, linenos);
  AnfNodePtrList node_inputs;
  AbstractBasePtrList abs_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    AnfNodePtr node;
    auto input_obj = inputs[i];
    bool is_parameter = py::hasattr(input_obj, "__parameter__") && py::isinstance<tensor::MetaTensor>(input_obj);
    // When the input of a cnode is a weight, add it to the top graph.
    if (is_parameter) {
      node = ConvertParameterObj(input_obj);
    } else {
      node = GetNode(inputs[i], debug_info);
      MS_EXCEPTION_IF_NULL(node);
    }
    (void)node_inputs.emplace_back(node);
    if (node->abstract() != nullptr) {
      (void)abs_inputs.emplace_back(node->abstract());
    } else {
      (void)abs_inputs.emplace_back(GetAbstract(input_obj));
    }
    MS_LOG(DEBUG) << "Add input, " << node->DebugString();
  }
  const auto &prim_py = std::make_shared<PrimitivePy>(prim_obj);
  AnfNodePtr cnode;
  if (py::cast<bool>(do_signature)) {
    cnode = prim::GenerateCNode(graph_stack_.top(), prim_py->name(), prim_py, abs_inputs, node_inputs);
  } else {
    cnode = GenerateCNode(graph_stack_.top(), prim_py, node_inputs);
  }
  if (cnode->debug_info() != nullptr) {
    cnode->debug_info()->set_trace_info(MakeTraceInfo<TraceOpt>(debug_info));
  }
  MS_LOG(DEBUG) << "New cnode: " << cnode->DebugString();
  if (GetPrimEffectInfo(prim_py).HasEffect()) {
    side_effect_nodes_.add(cnode);
    return;
  }
  SetNode(prim_res, cnode, debug_info);
}

AnfNodePtr TraceRecorder::GetNode(const py::object &obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  if (py::isinstance<tensor::MetaTensor>(obj)) {
    return GetTensorNode(obj, debug_info, set_abstract);
  } else if (py::isinstance<py::bool_>(obj)) {
    MS_LOG(DEBUG) << "Constant bool: " << py::str(obj);
    const auto &value_node = NewValueNode(py::cast<bool>(obj));
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<py::int_>(obj)) {
    MS_LOG(DEBUG) << "Constant int64_t: " << py::str(obj);
    const auto &value_node = NewValueNode(py::cast<int64_t>(obj));
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<py::float_>(obj)) {
    MS_LOG(DEBUG) << "Constant float: " << py::str(obj);
    auto data = py::cast<float>(obj);
    const auto &value_node = NewValueNode(data);
    auto fp32_val = value_node->value()->cast<FP32ImmPtr>();
    if (fp32_val != nullptr) {
      MS_LOG(DEBUG) << "Set float64 value to FP32Imm.";
      fp32_val->set_prim_value(py::cast<double>(obj));
    }
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<py::str>(obj)) {
    MS_LOG(DEBUG) << "Constant str: " << py::str(obj);
    const auto &value_node = NewValueNode(py::cast<std::string>(obj));
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<py::none>(obj)) {
    MS_LOG(DEBUG) << "Constant none: " << py::str(obj);
    const auto &value_node = NewValueNode(kNone);
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<py::ellipsis>(obj)) {
    MS_LOG(DEBUG) << "Constance ellipsis: " << py::str(obj);
    const auto &value_node = NewValueNode(kEllipsis);
    if (set_abstract) {
      value_node->set_abstract(GetAbstract(obj));
    }
    return value_node;
  } else if (py::isinstance<Type>(obj)) {
    MS_LOG(DEBUG) << "Constance type: " << py::str(obj);
    const auto &type_node = NewValueNode(obj.cast<TypePtr>());
    if (set_abstract) {
      type_node->set_abstract(GetAbstract(obj));
    }
    return type_node;
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &tuple_obj = py::cast<py::tuple>(obj);
    return GetTupleNode(tuple_obj, debug_info, set_abstract);
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &list_obj = py::cast<py::list>(obj);
    return GetListNode(list_obj, debug_info, set_abstract);
  }
  Clear();
  MS_LOG(INTERNAL_EXCEPTION) << "Not support [" << py::str(obj.get_type()) << "] " << py::str(obj)
                             << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
}

AnfNodePtr TraceRecorder::GetTensorNode(const py::object &tensor_obj, const DebugInfoPtr &debug_info,
                                        bool set_abstract) {
  const auto &tensor = tensor_obj.cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  // Get the preceding CNode firstly.
  if (tensor->has_user_data("__node__")) {
    const auto &node = tensor->user_data<AnfNode>("__node__");
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Get node from [" << py::str(tensor_obj.get_type()) << "] " << GetPyObjId(tensor_obj) << "/"
                 << py::str(tensor_obj) << ", ptr: " << tensor.get() << ", " << node->DebugString()
                 << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
    return node;
  }
  // No preceding node, create a ValueNode.
  MS_LOG(INFO) << "No preceding node found, to create tensor value node. [" << py::str(tensor_obj.get_type()) << "] "
               << GetPyObjId(tensor_obj) << "/" << py::str(tensor_obj) << ", ptr: " << tensor.get()
               << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
  const auto &value_node = NewValueNode(tensor);
  if (set_abstract) {
    value_node->set_abstract(GetAbstract(tensor_obj));
  }
  return value_node;
}

AnfNodePtr TraceRecorder::GetTupleNode(const py::tuple &tuple_obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  // Find the object firstly.
  const auto &obj_str = GetPyObjId(tuple_obj);
  MS_LOG(DEBUG) << "To find node by tuple, whose obj id: " << obj_str
                << ", tuple_obj: " << py::str(py::cast<py::object>(tuple_obj));
  auto iter = py_obj_node_map_.find(obj_str);
  if (iter != py_obj_node_map_.cend()) {
    MS_LOG(DEBUG) << "Found preceding node by tuple obj id: " << obj_str
                  << ", tuple_obj: " << py::str(py::cast<py::object>(tuple_obj))
                  << ", node: " << iter->second->DebugString();
    return iter->second;
  }
  // Create MakeTuple CNode.
  AnfNodePtrList make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < tuple_obj.size(); ++i) {
    const auto &node = GetNode(tuple_obj[i], debug_info);
    (void)make_tuple_inputs.emplace_back(node);
  }
  const auto &make_tuple_cnode = graph_stack_.top()->NewCNodeInOrder(make_tuple_inputs);
  if (set_abstract) {
    make_tuple_cnode->set_abstract(GetAbstract(tuple_obj));
  }
  py_obj_node_map_[obj_str] = make_tuple_cnode;
  MS_LOG(DEBUG) << "Not found preceding node by tuple obj id: " << obj_str
                << ", tuple_obj: " << py::str(py::cast<py::object>(tuple_obj))
                << ", new node: " << make_tuple_cnode->DebugString();
  return make_tuple_cnode;
}

AnfNodePtr TraceRecorder::GetListNode(const py::list &list_obj, const DebugInfoPtr &debug_info, bool set_abstract) {
  // Create MakeList CNode for py::list each time.
  AnfNodePtrList make_list_inputs = {NewValueNode(prim::kPrimMakeList)};
  for (size_t i = 0; i < list_obj.size(); ++i) {
    const auto &node = GetNode(list_obj[i], debug_info);
    (void)make_list_inputs.emplace_back(node);
  }
  const auto &list_cnode = graph_stack_.top()->NewCNodeInOrder(make_list_inputs);
  if (set_abstract) {
    list_cnode->set_abstract(GetAbstract(list_obj));
  }
  return list_cnode;
}

void TraceRecorder::SetNode(const py::object &obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                            bool set_abstract) {
  if (py::isinstance<tensor::MetaTensor>(obj)) {
    const auto &tensor = obj.cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_user_data<AnfNode>("__node__", node);
    MS_LOG(INFO) << "Set node to [" << py::str(obj.get_type()) << "] " << GetPyObjId(obj) << "/" << py::str(obj)
                 << ", ptr: " << tensor.get() << ", " << node->DebugString()
                 << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::bool_>(obj)) {
    MS_LOG(DEBUG) << "Constant bool: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::int_>(obj)) {
    MS_LOG(DEBUG) << "Constant int64_t: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::float_>(obj)) {
    MS_LOG(DEBUG) << "Constant float: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::str>(obj)) {
    MS_LOG(DEBUG) << "Constant str: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::none>(obj)) {
    MS_LOG(DEBUG) << "Constant none: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::ellipsis>(obj)) {
    MS_LOG(DEBUG) << "Constance ellipsis: " << py::str(obj);
    if (set_abstract) {
      node->set_abstract(GetAbstract(obj));
    }
    return;
  } else if (py::isinstance<py::tuple>(obj)) {
    const py::tuple &tuple_obj = py::cast<py::tuple>(obj);
    SetTupleNode(tuple_obj, node, debug_info, set_abstract);
    return;
  } else if (py::isinstance<py::list>(obj)) {
    const py::list &list_obj = py::cast<py::list>(obj);
    SetListNode(list_obj, node, debug_info, set_abstract);
    return;
  }
  Clear();
  MS_LOG(INTERNAL_EXCEPTION) << "Not support [" << py::str(obj.get_type()) << "] " << py::str(obj)
                             << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
}

void TraceRecorder::SetTupleNode(const py::tuple &tuple_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                                 bool set_abstract) {
  if (!IsMutable(tuple_obj)) {
    const auto &obj_str = GetPyObjId(tuple_obj);
    py_obj_node_map_[obj_str] = node;
    MS_LOG(DEBUG) << "Set node for tuple, whose obj id: " << obj_str << ", [" << py::str(tuple_obj.get_type()) << "] "
                  << py::str(py::cast<py::object>(tuple_obj)) << ", " << node->DebugString()
                  << ", line: " << trace::GetDebugInfoStr(debug_info, "", kSourceLineTipDiscard);
    if (set_abstract) {
      node->set_abstract(GetAbstract(tuple_obj));
    }
    // Not return, create tensor -> node relation by tuple items in advance.
  }
  const auto &prim = GetCNodePrimitive(node);
  // It's mutable tuple.
  for (size_t i = 0; i < tuple_obj.size(); ++i) {
    if (!IsMutable(tuple_obj) && !py::isinstance<tensor::MetaTensor>(tuple_obj[i])) {
      continue;
    }
    // Create Tuple GetItem CNode.
    AnfNodePtrList tuple_getitem_inputs = {NewValueNode(prim::kPrimTupleGetItem)};
    (void)tuple_getitem_inputs.emplace_back(node);
    (void)tuple_getitem_inputs.emplace_back(NewValueNode(SizeToLong(i)));
    const auto &getitem_cnode = graph_stack_.top()->NewCNodeInOrder(tuple_getitem_inputs);
    SetNode(tuple_obj[i], getitem_cnode, debug_info, set_abstract);
  }
}

void TraceRecorder::SetListNode(const py::list &list_obj, const AnfNodePtr &node, const DebugInfoPtr &debug_info,
                                bool set_abstract) {
  if (!IsMutable(list_obj)) {
    // Not create list -> node relation, to get node by GetNode(list_obj) if need.
    if (set_abstract) {
      node->set_abstract(GetAbstract(list_obj));
    }
    // Not return, create tensor -> node relation by list items in advance.
  }
  // It's mutable list.
  for (size_t i = 0; i < list_obj.size(); ++i) {
    // Create List GetItem CNode.
    AnfNodePtrList list_getitem_inputs = {NewValueNode(prim::kPrimListGetItem)};
    (void)list_getitem_inputs.emplace_back(node);
    (void)list_getitem_inputs.emplace_back(NewValueNode(SizeToLong(i)));
    const auto &getitem_cnode = graph_stack_.top()->NewCNodeInOrder(list_getitem_inputs);
    SetNode(list_obj[i], getitem_cnode, debug_info, set_abstract);
  }
}

void TraceRecorder::SyncTensorNode(const py::object &old_tensor_obj, const py::object &new_tensor_obj) {
  if (!py::isinstance<tensor::MetaTensor>(old_tensor_obj) || !py::isinstance<tensor::MetaTensor>(new_tensor_obj)) {
    return;
  }
  const auto &old_tensor = old_tensor_obj.cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(old_tensor);
  if (!old_tensor->has_user_data("__node__")) {
    MS_LOG(INTERNAL_EXCEPTION) << "Has no node in tensor, [" << py::str(old_tensor_obj.get_type()) << "] "
                               << py::str(old_tensor_obj) << ", ptr: " << old_tensor.get();
  }
  const auto &node = old_tensor->user_data<AnfNode>("__node__");
  MS_EXCEPTION_IF_NULL(node);
  const auto &new_tensor = new_tensor_obj.cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(new_tensor);
  new_tensor->set_user_data<AnfNode>("__node__", node);
  MS_LOG(DEBUG) << "Sync node from [" << py::str(old_tensor_obj.get_type()) << "] ptr: " << old_tensor.get() << " to ["
                << py::str(new_tensor_obj.get_type()) << "] ptr: " << new_tensor.get()
                << ", node: " << node->DebugString();
}

py::object TraceRecorder::InitTraceGraphInputs(const AbstractBasePtr &abs, const AnfNodePtr &param) {
  MS_EXCEPTION_IF_NULL(abs);
  auto val = abs->BuildValue();
  bool has_value = val != nullptr && !val->ContainsValueAny();
  if (abs->isa<abstract::AbstractSequence>()) {
    param->set_abstract(abs);
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>()->elements();
    py::tuple tuple_node(abs_seq.size());
    for (size_t i = 0; i < abs_seq.size(); ++i) {
      auto element = param->func_graph()->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), param, NewValueNode(SizeToLong(i))});
      tuple_node[i] = InitTraceGraphInputs(abs_seq[i], element);
    }
    return tuple_node;
  } else if (abs->isa<abstract::AbstractTensor>() || !has_value) {
    if (!abs->isa<abstract::AbstractTensor>()) {
      MS_LOG(WARNING) << "Input should be Tensor, but get " << abs->ToString() << ".";
    }
    param->set_abstract(abs);
    auto type_ptr = abs->GetType();
    MS_EXCEPTION_IF_NULL(type_ptr);
    auto tensor_type_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type_ptr);
    auto type_id = tensor_type_ptr->element()->type_id();
    auto shape_ptr = abs->GetShape();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto shape_vec = shape_ptr->GetShapeVector();
    auto tensor_ptr = std::make_shared<tensor::Tensor>(type_id, shape_vec);
    auto py_tensor = py::cast(tensor_ptr);
    SetNode(py_tensor, param, param->debug_info());
    return py_tensor;
  } else {
    param->set_abstract(abs);
    auto py_data = ValueToPyData(val);
    return py_data;
  }
}

void RegTraceRecorderPy(const py::module *m) {
  (void)py::class_<TraceRecorder, std::shared_ptr<TraceRecorder>>(*m, "TraceRecorder")
    .def_static("get_instance", &TraceRecorder::GetInstance, "Get trace manager instance.")
    .def("begin_graph", &TraceRecorder::BeginGraph, "Start a new graph.")
    .def("end_graph", &TraceRecorder::EndGraph, "Finish graph building.")
    .def("run_graph", &TraceRecorder::RunGraph, "Run the built graph.")
    .def("new_node", &TraceRecorder::NewNode, "Append a new CNode into current graph.")
    .def("sync_tensor_node", &TraceRecorder::SyncTensorNode, "Sync node from a tensor to another.")
    .def("new_fg_node", &TraceRecorder::NewFuncGraphNode, "Append a new CNode of func graph into current graph.");
}
}  // namespace trace
}  // namespace mindspore
