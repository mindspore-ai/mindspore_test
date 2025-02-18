/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pi/graph_compiler/compiler.h"
#include <memory>
#include <string>
#include <algorithm>
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/convert_utils_py.h"
#include "ir/func_graph.h"
#include "pipeline/jit/pi/graph_compiler/func_graph_builder.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include "pipeline/jit/ps/pipeline_jit.h"
#include "pipeline/pynative/pynative_execute.h"

namespace mindspore {
namespace pijit {
namespace {
// Reference : method _generate_run_args of _JitExecutor in api.py
// Parameters should be eliminated in the following case：
// 1.Constant Tensor, reason : constant folding
// 2.Constant Scalar(exclude those will be broaden), reason : constant folding
// 3.None, reason : reason : constant folding or not use
// 4.Empty constant length container(tuple/list/dict): constant folding or not use
// 5.Other(Graph Not Support)
bool IsValidRunArg(const py::object &obj, bool enable_tuple_broaden) {
  if (GraphUtils::IsTensor(obj)) {
    if (GraphUtils::HasInit(obj)) {
      (void)python_adapter::CallPyObjMethod(obj, "init_data");
    }
    return !GraphUtils::IsConst(obj);
  }
  // If the container input is empty and not variable length, graph treat it as constant, it should be erased in inputs.
  if (!GraphUtils::IsDynamicLength(obj) && GraphUtils::IsEmptyContainer(obj)) {
    return false;
  }
  return GraphUtils::IsMutable(obj) || GraphUtils::IsGradForScalar(obj) ||
         (enable_tuple_broaden && GraphUtils::IsTupleCanBroaden(obj));
}

bool CanbeMutable(const py::object &arg) {
  if (GraphUtils::IsConst(arg)) {
    return false;
  }
  // not necessary
  if (GraphUtils::IsMutable(arg)) {
    return false;
  }
  if (py::isinstance<py::dict>(arg) || py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    py::object o = python_adapter::CallPyFn("mindspore.common.mutable", "_check_element_type", arg);
    return py::isinstance<py::bool_>(o) && py::bool_(o);
  }
  return false;
}

void MarkArgumentMutable(const py::tuple &args) {
  for (size_t idx = 0; idx < args.size(); idx++) {
    if (CanbeMutable(args[idx])) {
      args[idx] = python_adapter::CallPyFn("mindspore.common", "mutable", args[idx]);
    }
  }
}

void MarkArgumentMutableWithParams(const py::tuple &args, const AnfNodePtrList &params) {
  for (auto param : params) {
    auto abstract = param->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (!abstract->isa<abstract::AbstractTensor>() && abstract->BuildValue() == kValueAny &&
        abstract->has_user_data("param_index")) {
      auto index = *(abstract->user_data<size_t>("param_index"));
      auto arg = args[index];
      if (GraphUtils::IsMutable(arg)) {
        continue;
      }
      py::object o = python_adapter::CallPyFn("mindspore.common.mutable", "_check_element_type", arg);
      if (py::isinstance<py::bool_>(o) && py::bool_(o)) {
        args[index] = python_adapter::CallPyFn("mindspore.common", "mutable", arg);
        MS_LOG(INFO) << "Add mutable to object index " << index;
      }
    }
  }
}

py::tuple MergeAllArgments(PyObject *args, PyObject *kwargs) {
  if (kwargs == nullptr) {
    return py::cast<py::tuple>(args);
  }
  py::list new_args;
  for (const auto &value : py::cast<py::tuple>(args)) {
    new_args.append(value);
  }
  for (const auto &[key, value] : py::cast<py::dict>(kwargs)) {
    (void)key;
    new_args.append(value);
  }
  return py::cast<py::tuple>(new_args);
}

py::tuple EliminateStubTensor(const py::tuple &args) {
  py::tuple new_args = py::reinterpret_steal<py::tuple>(PyTuple_New(args.size()));
  for (size_t idx = 0; idx < args.size(); idx++) {
    new_args[idx] = IsStubTensor(args[idx]) ? python_adapter::CallPyObjMethod(args[idx], "stub_sync") : args[idx];
  }
  return new_args;
}

py::tuple EliminateSelf(const py::tuple &args, const std::string &name) {
  if (!args.empty() && !GraphUtils::IsTensor(args[0]) && py::hasattr(args[0], common::SafeCStr(name))) {
    return py::reinterpret_steal<py::tuple>(PyTuple_GetSlice(args.ptr(), 1, args.size()));
  }
  return args;
}

py::tuple EliminateInvalidArgs(const py::tuple &args, int co_flags, bool enable_tuple_broaden) {
  py::list new_args;
  for (size_t idx = 0; idx < args.size(); idx++) {
    if (IsValidRunArg(args[idx], enable_tuple_broaden)) {
      if ((idx < (args.size() - 1) || (IntToSize(co_flags) & CO_VARKEYWORDS) == 0) &&
          py::isinstance<py::dict>(args[idx])) {
        new_args.append(py::reinterpret_steal<py::tuple>(PyDict_Values(args[idx].ptr())));
      } else {
        new_args.append(args[idx]);
      }
    }
  }
  return py::cast<py::tuple>(new_args);
}

py::tuple ExpandVariableArgs(const py::tuple &args, int co_flags, int co_argcount) {
  if ((IntToSize(co_flags) & CO_VARARGS) == 0x0) {
    return args;
  }
  py::tuple var_args = py::cast<py::tuple>(args[co_argcount]);
  py::list new_args;
  for (int index = 0; index < co_argcount; index++) {
    new_args.append(args[index]);
  }
  for (const auto &var_arg : var_args) {
    new_args.append(var_arg);
  }
  for (size_t index = (size_t)co_argcount + 1; index < args.size(); index++) {
    new_args.append(args[index]);
  }
  return py::cast<py::tuple>(new_args);
}

PyObject *RunGraph(const std::string &phase, const py::tuple &args, const std::string &name, int co_flags,
                   bool enable_tuple_broaden) {
  auto graph_executor = pipeline::GetExecutor();
  MS_EXCEPTION_IF_NULL(graph_executor);
  py::tuple args_tuple = EliminateSelf(args, name);
  args_tuple = EliminateStubTensor(args_tuple);
  auto origin_fg = graph_executor->GetFuncGraph(phase);
  const auto &params = origin_fg->parameters();
  MarkArgumentMutableWithParams(args_tuple, params);
  MarkArgumentMutable(args_tuple);
  args_tuple = EliminateInvalidArgs(args_tuple, co_flags, enable_tuple_broaden);
  MS_LOG(INFO) << "Args for run: " << std::string(py::str(args_tuple));
  py::object ret;
  int mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
  auto executor = pynative::PyNativeExecutor::GetInstance();
  if (mode == kPynativeMode && executor->RequiresGrad()) {
    MS_LOG(INFO) << "Do GradJit";
    executor->grad_executor()->jit()->set_graph_phase(phase);
    ret = executor->GradJit(args_tuple);
  } else {
    ret = graph_executor->Run(args_tuple, py::str(phase));
  }
  FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  ret = python_adapter::CallPyFn("mindspore.common.api", "_convert_python_data", ret);
  ret.inc_ref();
  return ret.ptr();
}

class SkipBoostInferScope {
 public:
  SkipBoostInferScope() {
    MS_LOG(DEBUG) << "Disable boost-infer when running PIJit with two-stages mode";
    origin_value_ = common::GetEnv("MS_DEV_BOOST_INFER");
    common::SetEnv("MS_DEV_BOOST_INFER", "0");
  }
  ~SkipBoostInferScope() { common::SetEnv("MS_DEV_BOOST_INFER", origin_value_.c_str()); }

 private:
  std::string origin_value_;
};
}  // namespace

CallableGraph Compiler::Compile(const PyFunctionObject &func, const PyFrameWrapper &frame, const std::string &phase) {
  const PyCodeObject *code = frame.GetCode().ptr();
  std::string name = py::cast<std::string>(code->co_name);
  MS_EXCEPTION_IF_CHECK_FAIL(!phase.empty(), "Phase name should not be empty for function " + name + ".");

  PyObject *f = reinterpret_cast<PyObject *>(const_cast<PyFunctionObject *>(&func));
  bool enable_tuple_broaden = GraphUtils::IsTupleBroadenEnable(py::cast<py::object>(f));
  CallableGraph callable = [code, enable_tuple_broaden, phase](PyObject *args, PyObject *kwargs) -> PyObject * {
    MS_EXCEPTION_IF_CHECK_FAIL(PyTuple_Check(args), "Excepted a Tuple Object for run args.");
    MS_EXCEPTION_IF_CHECK_FAIL(((kwargs == nullptr) || PyDict_Check(kwargs)),
                               "Excepted nullptr or a Dict Object for run kwargs.");

    py::tuple tuple = MergeAllArgments(args, kwargs);
    tuple = ExpandVariableArgs(tuple, code->co_flags, code->co_argcount);
    std::string name = py::cast<std::string>(code->co_name);
    return RunGraph(phase, tuple, name, code->co_flags, enable_tuple_broaden);
  };

  auto graph_executor = mindspore::pipeline::GraphExecutorPy::GetInstance();
  if (graph_executor->HasCompiled(phase)) {
    return callable;
  }

  int arg_cnt = code->co_argcount + code->co_kwonlyargcount;
  if (IntToSize(code->co_flags) & CO_VARARGS) {
    arg_cnt++;
  }
  py::dict f_locals = frame.Locals();
  py::list locals = py::reinterpret_steal<py::list>(PyDict_Values(f_locals.ptr()));
  py::tuple args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(PyList_GetSlice(locals.ptr(), 0, arg_cnt)));
  py::dict kwargs =
    (IntToSize(code->co_flags) & CO_VARKEYWORDS) == 0x0 ? py::dict() : py::cast<py::dict>(locals[arg_cnt]);
  args = EliminateStubTensor(args);
  auto byteCodeParser = std::make_shared<ByteCodeParser>(func);
  ir::FunctionNodePtr func_node = byteCodeParser->Parse();
  FuncGraphPtr graph = FuncGraphBuilder::BuildFuncGraph(func_node, args, kwargs);
  if (graph == nullptr) {
    return nullptr;
  }
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("func_graph_builder.ir", graph);
  }
  args = ExpandVariableArgs(args, code->co_flags, code->co_argcount);
  MarkArgumentMutable(args);
  try {
    SkipBoostInferScope skip_boost_infer_scope;
    (void)graph_executor->CompileInner(graph, args, kwargs, phase, false);
  } catch (const std::exception &ex) {
    MS_LOG(ERROR) << "CompileInner failed for [" << std::string(py::str(name)) << "], error:" << ex.what();
    return nullptr;
  }
  return callable;
}

namespace {
py::tuple MergeArgsKwargs(PyObject *args, PyObject *kwargs) {
  if (kwargs == nullptr) {
    return py::cast<py::tuple>(args);
  }
  py::list new_args;
  for (const auto &value : py::cast<py::tuple>(args)) {
    new_args.append(value);
  }
  // Graph mode will convert dict input to tuple with values.
  py::list converted_kwargs;
  for (const auto &[key, value] : py::cast<py::dict>(kwargs)) {
    (void)key;
    converted_kwargs.append(value);
  }
  if (py::len(converted_kwargs) != 0) {
    new_args.append(py::cast<py::tuple>(converted_kwargs));
  }
  return py::cast<py::tuple>(new_args);
}
}  // namespace

CallableGraph MindCompiler::Compile(const FuncGraphPtr &func_graph, const py::tuple &args, const py::dict &kwargs,
                                    const std::string &phase, const CompileInfo &compile_info) {
  MS_EXCEPTION_IF_CHECK_FAIL(!phase.empty(),
                             "Phase name should not be empty for function " + compile_info.co_name_ + ".");
  CallableGraph callable = [compile_info, phase](PyObject *args, PyObject *kwargs) -> PyObject * {
    MS_EXCEPTION_IF_CHECK_FAIL(PyTuple_Check(args), "Excepted a Tuple Object for run args.");
    MS_EXCEPTION_IF_CHECK_FAIL(((kwargs == nullptr) || PyDict_Check(kwargs)),
                               "Excepted nullptr or a Dict Object for run kwargs.");

    py::tuple tuple = MergeArgsKwargs(args, kwargs);
    return RunGraph(phase, tuple, compile_info.co_name_, compile_info.co_flags_, false);  // need adapt for optimizer
  };

  auto jit_executor = pipeline::GetExecutor();
  if (jit_executor->HasCompiled(phase)) {
    return callable;
  }

  if (func_graph == nullptr) {
    return nullptr;
  }
  py::tuple new_arg = EliminateStubTensor(args);
  new_arg = EliminateSelf(new_arg, compile_info.co_name_);
  MarkArgumentMutable(new_arg);
  if (MsContext::GetInstance()->CanDump(kIntroductory)) {
    DumpIR("graph_before_compile.ir", func_graph);
  }
  MS_LOG(INFO) << "Args for compile: " << std::string(py::str(new_arg));

  auto origin_num = compile_info.origin_top_input_num_;

  const auto &params = func_graph->parameters();
  if (origin_num != params.size()) {
    MS_LOG(INFO) << "Reorder top function graph inputs.";
    AnfNodePtrList new_params;
    (void)std::copy(params.begin(), params.begin() + origin_num, std::back_inserter(new_params));
    (void)std::copy_if(params.begin() + origin_num, params.end(), std::back_inserter(new_params),
                       [](const AnfNodePtr &param) { return !param->abstract()->isa<abstract::AbstractRefTensor>(); });
    (void)std::copy_if(params.begin() + origin_num, params.end(), std::back_inserter(new_params),
                       [](const AnfNodePtr &param) { return param->abstract()->isa<abstract::AbstractRefTensor>(); });
    func_graph->set_parameters(new_params);
  }

  (void)jit_executor->CompileInner(func_graph, new_arg, kwargs, phase, true);

  return callable;
}
}  // namespace pijit
}  // namespace mindspore
