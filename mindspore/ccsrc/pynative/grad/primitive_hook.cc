/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pynative/grad/primitive_hook.h"
#include <memory>
#include <string>
#include "include/common/utils/primitive_utils.h"
#include "pynative/pynative_execute.h"

namespace mindspore {
namespace {
constexpr auto kCellIDAttrName = "cell_id";
constexpr auto kCustomOpNameAttrName = "custom_op_name";
constexpr auto kIsRecomputeAttr = "is_recompute";

py::tuple UnfoldPyArgs(const py::tuple &py_args) {
  py::list list;
  for (size_t i = 0; i < py_args.size(); i++) {
    list = list + py::cast<py::list>(py_args[i]);
  }
  return py::cast<py::tuple>(list);
}
}  // namespace
BaseRef RunCellCustomBpropFunction(const PrimitivePyPtr &self, const py::tuple &py_args) {
  py::tuple converted_args = ConvertCTensorToPyTensor(py_args);
  MS_LOG(DEBUG) << "Get convert args size " << converted_args.size() << ", args are "
                << ConvertPyObjToString(converted_args);
  // If recompute, just discard dout; Otherwise, discat out and dout
  bool is_recompute = self->HasAttr(kIsRecomputeAttr);
  size_t non_inp_args_size = is_recompute ? kSizeOne : kSizeTwo;

  auto inp_args_size = py_args.size() - non_inp_args_size;
  py::tuple input_args(inp_args_size);
  for (size_t i = 0; i < inp_args_size; ++i) {
    input_args[i] = py_args[i];
  }
  MS_LOG(DEBUG) << "Get cell input arg size " << inp_args_size;
  // Run bprop function.
  const auto &inst = pynative::PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  try {
    MS_LOG(DEBUG) << "Run cell custom bprop function start.";
    py::tuple grads;
    if (inst->grad_flag()) {
      inst->NewGraph(self->hook_fn(), input_args.cast<py::args>());
    }
    py::object grads_obj = self->hook_fn()(*converted_args);
    MS_LOG(DEBUG) << "Get cell hook output " << ConvertPyObjToString(grads_obj);
    grads = CheckBpropOut(grads_obj, py_args, self->bprop_cls_name());
    py::object out = grads_obj;
    // If grads.size() > inp_args_size, that means exist weights.
    if (grads.size() > inp_args_size) {
      MS_LOG(DEBUG) << "Get grads size " << grads.size();
      out = py::cast<py::tuple>(grads_obj)[0];
    }
    if (inst->grad_flag()) {
      inst->EndGraph(self->hook_fn(), out, input_args.cast<py::args>());
    }
    MS_LOG(DEBUG) << "Run cell custom bprop function end.";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef RunCustomOpBpropFunction(const PrimitivePyPtr &self, const py::tuple &ori_py_args) {
  auto is_custom_aot_node = self->HasAttr(kCustomOpNameAttrName) &&
                            GetValue<std::string>(self->GetAttr(kCustomOpNameAttrName)) == kCustomExtOpName;
  MS_LOG(DEBUG) << "Custom op:" << self->name() << ", is custom aot node: " << is_custom_aot_node;
  py::tuple py_args = is_custom_aot_node ? UnfoldPyArgs(ori_py_args) : ori_py_args;
  py::tuple converted_args = ConvertCTensorToPyTensor(py_args);
  MS_LOG(DEBUG) << "Get convert args size " << converted_args.size() << ", args are "
                << ConvertPyObjToString(converted_args);
  try {
    MS_LOG(DEBUG) << "Run custom op bprop start";
    py::object grads_obj = self->hook_fn()(*converted_args);
    auto grads = CheckBpropOut(grads_obj, py_args, self->bprop_cls_name());
    MS_LOG(DEBUG) << "Run custom op bprop end";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef RunCellHookFunction(const PrimitivePyPtr &self, const py::tuple &py_args) {
  const auto args_size = py_args.size();
  // Get the din passed to current bprop cut op.
  py::object grad_output = py_args[args_size - 1];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  try {
    const auto &hook_fn = self->hook_fn();
    const auto &hook_type = self->hook_type();
    MS_LOG(DEBUG) << "Get cell dout " << ConvertPyObjToString(grad_output);
    if (self->hook_type() == HookType::kBackwardPreHook) {
      MS_LOG(DEBUG) << "Run cell backward pre hook function start.";
      py::object ret = hook_fn(grad_output);
      if (!py::isinstance<py::none>(ret)) {
        MS_LOG(DEBUG) << "Get cell backward pre hook new grad output " << ConvertPyObjToString(ret);
        const auto &code_obj = py::getattr(hook_fn, "__code__");
        py::object co_name = py::getattr(code_obj, "co_name");
        self->CheckHookConsistency(self->UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
        grad_output = ret;
      }
      MS_LOG(DEBUG) << "Run cell backward pre hook function end.";
      return std::make_shared<PyObjectRef>(grad_output);
    }
    if (hook_type == HookType::kBackwardHook) {
      MS_LOG(DEBUG) << "Run cell backward hook function start.";
      py::object ret = hook_fn(grad_output);
      if (py::isinstance<py::str>(ret)) {
        MS_LOG(DEBUG) << "Run cell " << ret.cast<std::string>() << " backward hook function the first time";
        self->EmplaceUnpairBackwardHookGrad(ret.cast<std::string>(), hook_fn);
        return std::make_shared<PyObjectRef>(grad_output);
      }
      if (py::isinstance<py::none>(ret)) {
        MS_LOG(DEBUG) << "Run cell backward hook function the second time with return None.";
      } else {
        MS_LOG(DEBUG) << "Get cell backward hook new grad input " << ConvertPyObjToString(ret);
        const auto &code_obj = py::getattr(hook_fn, "__code__");
        py::object co_name = py::getattr(code_obj, "co_name");
        self->CheckHookConsistency(self->UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
        grad_output = ret;
      }
      self->EraseUnpairBackwardHookGrad(GetValue<std::string>(self->GetAttr(kCellIDAttrName)));
      MS_LOG(DEBUG) << "Run cell backward hook function end.";
      return std::make_shared<PyObjectRef>(grad_output);
    }
    MS_LOG(EXCEPTION) << "Get unsupported hook function type";
  } catch (std::exception &bt) {
    self->ClearUnpairBackwardHookGrad();
    auto inst = pynative::PyNativeExecutor::GetInstance();
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef RunVariableHookFunction(const PrimitivePyPtr &self, const py::tuple &py_args) {
  constexpr size_t grad_output_index = 0;
  if (py_args.size() != kSizeOne) {
    MS_LOG(EXCEPTION) << "Bprop cut run must in the following format: dout";
  }
  py::object grad_output = py_args[grad_output_index];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  MS_LOG(DEBUG) << "Get grad output " << ConvertPyObjToString(grad_output);
  try {
    const auto &hook_type = self->hook_type();
    const auto &hook_fn = self->hook_fn();
    if (hook_type == HookType::kUnknown) {
      MS_LOG(EXCEPTION) << "Get unsupported hook type Unknown";
    }
    if (hook_type == HookType::kTensorHook) {
      MS_LOG(DEBUG) << "Run tensor hook function begin";
    } else {
      // Op maybe have multi outputs, so wrap to tuple for unitary.
      // Tensor hook just work on tensor, so keep origin input style
      if (!py::isinstance<py::tuple>(grad_output)) {
        grad_output = py::make_tuple(grad_output);
      }
      MS_LOG(DEBUG) << "Run HookBackward op function begin";
    }
    auto ret = hook_fn(grad_output);
    if (!py::isinstance<py::none>(ret)) {
      MS_LOG(DEBUG) << "Get hook output " << ConvertPyObjToString(ret);
      grad_output = ret;
    }

    if (hook_type != HookType::kTensorHook) {
      const auto &code_obj = py::getattr(hook_fn, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      self->CheckHookConsistency(self->UnpackRetValueOfCellHook(grad_output), py_args[grad_output_index], co_name);
    }
    MS_LOG(DEBUG) << (hook_type == HookType::kTensorHook ? "Run tensor hook function end"
                                                         : "Run HookBackward op function end");
    if (!py::isinstance<py::tuple>(grad_output)) {
      grad_output = py::make_tuple(grad_output);
    }
    return std::make_shared<PyObjectRef>(grad_output);
  } catch (std::exception &bt) {
    auto inst = pynative::PyNativeExecutor::GetInstance();
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef RunHookFunction(const PrimitivePyPtr &self, const VectorRef &args) {
  py::tuple py_args = ConvertDatatoPyTuple(args);
  MS_LOG(DEBUG) << "Get input args size " << py_args.size() << ", args are " << ConvertPyObjToString(py_args);
  // For cell has custom bprop function
  if (self->hook_type() == HookType::kCellCustomBprop) {
    MS_LOG(DEBUG) << "Run cell custom bprop";
    return RunCellCustomBpropFunction(self, py_args);
  }

  // For cell register hook
  if (self->hook_type() == HookType::kBackwardPreHook || self->hook_type() == HookType::kBackwardHook) {
    MS_LOG(DEBUG) << "Run cell backward hook";
    return RunCellHookFunction(self, py_args);
  }

  // For custom op, which define custrcut and bprop
  if (self->hook_type() == HookType::kCustomOpBprop) {
    MS_LOG(DEBUG) << "Run custom op";
    return RunCustomOpBpropFunction(self, py_args);
  }

  // For hook use, include hook op and tensor register hook
  return RunVariableHookFunction(self, py_args);
}

struct RunPrimitivePyHookFunctionRegister {
  RunPrimitivePyHookFunctionRegister() {
    python_adapter::PyAdapterCallback::SetRunPrimitivePyHookFunctionHandler(
      [](const PrimitivePtr &prim, const VectorRef &args) -> BaseRef {
        auto py_prim = prim->cast<PrimitivePyPtr>();
        MS_EXCEPTION_IF_NULL(py_prim);
        return RunHookFunction(py_prim, args);
      });
  }
} callback_register;
struct ProcessUnPairedCellHookRegister {
  ProcessUnPairedCellHookRegister() {
    python_adapter::PyAdapterCallback::SetProcessUnPairedCellHookHandler(
      [](bool execute_hook_fn) -> void { PrimitivePy::ProcessUnPairedCellHook(execute_hook_fn); });
  }
} cell_hook_callback_register;
}  // namespace mindspore
