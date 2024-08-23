/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pybind_api/ir/primitive_py.h"

#include <map>
#include "ir/signature.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/utils/python_adapter.h"
#include "pybind11/pytypes.h"
#include "include/common/pybind_api/api_register.h"
#include "pybind_api/export_flags.h"
#include "pybind_api/ir/base_ref_py.h"
#include "utils/convert_utils_base.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "include/common/utils/primitive_utils.h"
#include "utils/check_convert_utils.h"
#include "pipeline/pynative/pynative_execute.h"
#include "include/common/profiler.h"

namespace mindspore {
namespace {
constexpr auto kCellIDAttrName = "cell_id";
constexpr auto kIsRecomputeAttr = "is_recompute";

static uint64_t MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}
std::map<std::string, std::string> kOpAttrNameReplaceMap = {
  {"data_format", "format"},
};

void SyncData(const py::object &arg) {
  if (py::isinstance<py::tuple>(arg)) {
    py::tuple arg_list = py::cast<py::tuple>(arg);
    for (size_t i = 0; i < arg_list.size(); i++) {
      SyncData(arg_list[i]);
    }
  }
  if (py::isinstance<tensor::Tensor>(arg)) {
    auto tensor = py::cast<tensor::TensorPtr>(arg);
    tensor->data_sync();
  }
  if (IsStubTensor(arg)) {
    auto tensor = ConvertStubTensor(arg);
    tensor->data_sync();
  }
}

bool ContainsWeights(const py::tuple &grads) {
  if (grads.size() < kSizeTwo) {
    return false;
  }
  if (!py::isinstance<py::tuple>(grads[0]) && !py::isinstance<py::dict>(grads[1])) {
    return false;
  }
  return true;
}

struct RunPrimitivePyHookFunctionRegister {
  RunPrimitivePyHookFunctionRegister() {
    python_adapter::PyAdapterCallback::SetRunPrimitivePyHookFunctionHandler(
      [](const PrimitivePtr &prim, const VectorRef &args) -> BaseRef {
        auto py_prim = prim->cast<PrimitivePyPtr>();
        MS_EXCEPTION_IF_NULL(py_prim);
        return py_prim->RunHookFunction(args);
      });
  }
} callback_register;
struct ProcessUnPairedCellHookRegister {
  ProcessUnPairedCellHookRegister() {
    python_adapter::PyAdapterCallback::SetProcessUnPairedCellHookHandler(
      [](bool execute_hook_fn) -> void { PrimitivePy::ProcessUnPairedCellHook(execute_hook_fn); });
  }
} cell_hook_callback_register;

std::map<HookType, std::string> hook_type_with_str = {
  {HookType::kCustomOpBprop, "CustomOpBprop"},
  {HookType::kCellCustomBprop, "CellCustomBprop"},
  {HookType::kHookBackwardOp, "HookBackwardOp"},
  {HookType::kTensorHook, "TensorHook"},
  {HookType::kBackwardPreHook, "BackwardPreHook"},
  {HookType::kBackwardHook, "BackwardHook"},
  {HookType::kUnknown, "Unknown"},
};
}  // namespace

std::unordered_map<std::string, py::function> PrimitivePy::unpair_backward_hook_grad_{};

PrimitivePy::PrimitivePy(const std::string &name) : Primitive(name, false), python_obj_(py::none()) {}

PrimitivePy::PrimitivePy(const PrimitivePy &prim_py)
    : Primitive(prim_py),
      python_obj_(prim_py.python_obj_),
      bprop_cls_name_(prim_py.bprop_cls_name_),
      adapter_(prim_py.adapter_),
      signatures_(prim_py.signatures_),
      bprop_cut_prims_(prim_py.bprop_cut_prims_),
      hook_type_(prim_py.hook_type_),
      hook_fn_(prim_py.hook_fn_) {}

PrimitivePy &PrimitivePy::operator=(const PrimitivePy &other) {
  if (this == &other) {
    return *this;
  }
  Primitive::operator=(other);
  python_obj_ = other.python_obj_;
  bprop_cls_name_ = other.bprop_cls_name_;
  adapter_ = other.adapter_;
  signatures_ = other.signatures_;
  bprop_cut_prims_ = other.bprop_cut_prims_;
  hook_fn_ = other.hook_fn_;
  hook_type_ = other.hook_type_;
  return *this;
}

PrimitivePy::PrimitivePy(const py::object &python_obj)
    : Primitive(python_obj.cast<PrimitivePyAdapterPtr>()->name_, false),
      python_obj_(python_obj),
      adapter_(python_obj.cast<PrimitivePyAdapterPtr>()) {
  MS_LOG(DEBUG) << "New primitive:" << adapter_->name_;
  set_signatures(adapter_->signatures_);
  (void)Primitive::SetAttrs(adapter_->attrs_);
  Primitive::set_prim_type(adapter_->prim_type_);
  Primitive::set_const_prim(adapter_->const_prim_);
  Primitive::set_inplace_prim(adapter_->inplace_prim_);
  Primitive::set_const_input_indexes(adapter_->const_input_indexes_);
  SetHookFn(adapter_->hook_fn_, adapter_->hook_type_);
  set_instance_name(adapter_->instance_name_);
  CloneUserData(adapter_->user_data_);
}

PrimitivePy::~PrimitivePy() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kDefault, name(),
                                     false);
  py::gil_scoped_acquire acquire_gil;
  python_obj_ = py::none();
  hook_fn_ = py::none();
}

py::function PrimitivePy::GetVmapRuleFunction(const bool, int axis_size) {
  constexpr char get_vmap_rule_func_name[] = "get_vmap_rule";
  if (py::hasattr(python_obj_, get_vmap_rule_func_name)) {
    return python_obj_.attr(get_vmap_rule_func_name)().cast<py::function>();
  }
  return GetVmapRuleFunctionByObj(python_obj_, axis_size);
}

py::function PrimitivePy::GetBpropFunction() {
  static const char *const get_bprop_func_name = "get_bprop";
  if (py::hasattr(python_obj_, get_bprop_func_name)) {
    py::function fn = python_obj_.attr(get_bprop_func_name)().cast<py::function>();
    return fn;
  }

  auto fn = GetBpropFunctionByObj(python_obj_);
  return fn;
}

py::function PrimitivePy::GetTaylorRuleFunction() {
  static const char *const get_taylor_rule_func_name = "get_taylor_rule";
  if (py::hasattr(python_obj_, get_taylor_rule_func_name)) {
    py::function fn = python_obj_.attr(get_taylor_rule_func_name)().cast<py::function>();
    return fn;
  }
  auto fn = GetTaylorRuleFunctionByObj(python_obj_);
  return fn;
}

void check_bprop_input_grads(const py::tuple &py_args, const py::tuple &grads, const std::string &bprop_cls_name,
                             int filter_args_size) {
  if (!MsContext::GetInstance()->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG)) {
    return;
  }
  if (grads.size() != py_args.size() - filter_args_size) {
    MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name
                            << "', the number of return values(gradients) should be equal to the number of input "
                               "arguments except 'out' and 'dout', which is: "
                            << (py_args.size() - filter_args_size) << ", but got:" << grads.size() << ".";
  }
  for (size_t i = 0; i < grads.size(); i++) {
    if (py::isinstance<tensor::Tensor>(py_args[i]) || IsStubTensor(py_args[i])) {
      if (!py::isinstance<tensor::Tensor>(grads[i]) && !IsStubTensor(grads[i])) {
        MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                << "th return value(gradient of the " << i << "th argument) should be Tensor, but got "
                                << py::cast<std::string>(grads[i].attr("__class__").attr("__name__"))
                                << ", and the value is " << py::cast<py::str>(grads[i]) << ".";
      }

      py::object arg_dtype = py_args[i].attr("dtype");
      py::object grad_dtype = grads[i].attr("dtype");
      py::tuple arg_shape = py_args[i].attr("shape");
      py::tuple grad_shape = grads[i].attr("shape");
      if (!grad_dtype.equal(arg_dtype)) {
        MS_EXCEPTION(TypeError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                << "th return value(gradient of the " << i
                                << "th argument) should have the same dtype as the " << i
                                << "th argument, which is:" << py::cast<py::str>(arg_dtype)
                                << ", but got: " << py::cast<py::str>(grad_dtype) << ".";
      }
      if (!grad_shape.equal(arg_shape)) {
        MS_EXCEPTION(ValueError) << "For user defined method 'bprop' of net '" << bprop_cls_name << "', the " << i
                                 << "th return value(gradient of the " << i
                                 << "th argument) should have the same shape as the " << i
                                 << "th argument, which is:" << py::cast<py::str>(arg_shape)
                                 << ", but got: " << py::cast<py::str>(grad_shape) << ".";
      }
    }
  }
}

py::tuple check_bprop_out(const py::object &grads_obj, const py::tuple &py_args, const std::string &bprop_cls_name) {
  py::tuple grads;
  if (py::isinstance<py::none>(grads_obj)) {
    MS_EXCEPTION(TypeError) << "The python function output is none.";
  } else if (!py::isinstance<py::tuple>(grads_obj)) {
    MS_LOG(DEBUG) << "Wrap a tuple";
    grads = py::make_tuple(grads_obj);
  } else {
    grads = py::cast<py::tuple>(grads_obj);
  }
  if (ContainsWeights(grads)) {
    MS_LOG(DEBUG) << "Contain weights";
    py::tuple input_grads = py::cast<py::tuple>(grads[0]);
    py::dict weight_grads = py::cast<py::dict>(grads[1]);
    check_bprop_input_grads(py_args, input_grads, bprop_cls_name, 1);
    if (weight_grads.empty()) {
      return input_grads;
    }
    py::tuple all_grads(input_grads.size() + weight_grads.size());
    for (size_t i = 0; i < input_grads.size(); ++i) {
      all_grads[i] = input_grads[i];
    }
    size_t i = 0;
    for (auto weight_grad : weight_grads) {
      all_grads[i + input_grads.size()] = weight_grad.second;
      ++i;
    }
    return all_grads;
  } else {
    MS_LOG(DEBUG) << "Not contain weights";
    check_bprop_input_grads(py_args, grads, bprop_cls_name, kSizeTwo);
    return grads;
  }
}

void PrimitivePy::AddBpropCutPrim(const PrimitivePyPtr &bprop_cut_prim) {
  MS_EXCEPTION_IF_NULL(bprop_cut_prim);
  (void)bprop_cut_prims_.emplace_back(bprop_cut_prim);
}

void PrimitivePy::SetHookFn(const py::function &hook_fn, HookType hook_type) {
  hook_fn_ = hook_fn;
  hook_type_ = hook_type;
  for (const auto &elem : bprop_cut_prims_) {
    PrimitivePyPtr bprop_cut_prim = elem.lock();
    if (bprop_cut_prim != nullptr) {
      bprop_cut_prim->SetHookFn(hook_fn, hook_type);
    }
  }
}

py::object PrimitivePy::UnpackRetValueOfCellHook(const py::object &grad_out) const {
  if (!py::isinstance<py::tuple>(grad_out)) {
    return grad_out;
  }
  auto out_tuple = py::cast<py::tuple>(grad_out);
  if (out_tuple.size() == 1) {
    // The input number of current cell is 1.
    return out_tuple[0];
  }
  return grad_out;
}

void PrimitivePy::CheckHookConsistency(const py::object &grad_out, const py::object &expected_grad_out,
                                       const py::object &co_name) const {
  if (py::isinstance<py::tuple>(expected_grad_out)) {
    if (!py::isinstance<py::tuple>(grad_out)) {
      MS_EXCEPTION(TypeError) << "The output gradient should be a tuple!";
    }
    auto actual_out_tuple = py::cast<py::tuple>(grad_out);
    auto expected_out_tuple = py::cast<py::tuple>(expected_grad_out);
    if (actual_out_tuple.size() != expected_out_tuple.size()) {
      MS_EXCEPTION(ValueError) << "The tuple size of output gradient should be " << expected_out_tuple.size()
                               << ", but it is " << actual_out_tuple.size();
    }
    for (size_t i = 0; i < expected_out_tuple.size(); ++i) {
      CheckHookConsistency(actual_out_tuple[i], expected_out_tuple[i], co_name);
    }
  }

  if (py::isinstance<tensor::Tensor>(expected_grad_out) || IsStubTensor(expected_grad_out)) {
    if (!py::isinstance<tensor::Tensor>(grad_out) && !IsStubTensor(grad_out)) {
      MS_EXCEPTION(TypeError) << "The output type of function: " << py::str(co_name) << " should be a tensor but got "
                              << py::cast<std::string>(grad_out.attr("__class__").attr("__name__")) << ".";
    }
    tensor::TensorPtr actual_out_tensor =
      IsStubTensor(grad_out) ? ConvertStubTensor(grad_out) : py::cast<tensor::TensorPtr>(grad_out);
    tensor::TensorPtr expected_out_tensor = IsStubTensor(expected_grad_out)
                                              ? ConvertStubTensor(expected_grad_out)
                                              : py::cast<tensor::TensorPtr>(expected_grad_out);
    MS_EXCEPTION_IF_NULL(actual_out_tensor);
    MS_EXCEPTION_IF_NULL(expected_out_tensor);
    if (actual_out_tensor->GetShapeAndDataTypeInfo() != expected_out_tensor->GetShapeAndDataTypeInfo()) {
      MS_EXCEPTION(ValueError) << "The output type of function: " << py::str(co_name)
                               << " is not consistent with the expected, it should be "
                               << expected_out_tensor->GetShapeAndDataTypeInfo() << ", but got "
                               << actual_out_tensor->GetShapeAndDataTypeInfo();
    }
  }
}

BaseRef PrimitivePy::RunCellCustomBpropFunction(const py::tuple &py_args) const {
  py::tuple converted_args = ConvertCTensorToPyTensor(py_args);
  MS_LOG(DEBUG) << "Get convert args size " << converted_args.size() << ", args are "
                << ConvertPyObjToString(converted_args);
  // If recompute, just discard dout; Otherwise, discat out and dout
  bool is_recompute = HasAttr(kIsRecomputeAttr);
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
      inst->NewGraph(hook_fn_, input_args.cast<py::args>());
    }
    py::object grads_obj = hook_fn_(*converted_args);
    MS_LOG(DEBUG) << "Get cell hook output " << ConvertPyObjToString(grads_obj);
    grads = check_bprop_out(grads_obj, py_args, bprop_cls_name_);
    py::object out = grads_obj;
    // If grads.size() > inp_args_size, that means exist weights.
    if (grads.size() > inp_args_size) {
      MS_LOG(DEBUG) << "Get grads size " << grads.size();
      out = py::cast<py::tuple>(grads_obj)[0];
    }
    if (inst->grad_flag()) {
      inst->EndGraph(hook_fn_, out, input_args.cast<py::args>());
    }
    MS_LOG(DEBUG) << "Run cell custom bprop function end.";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunCustomOpBpropFunction(const py::tuple &py_args) const {
  py::tuple converted_args = ConvertCTensorToPyTensor(py_args);
  MS_LOG(DEBUG) << "Get convert args size " << converted_args.size() << ", args are "
                << ConvertPyObjToString(converted_args);
  try {
    MS_LOG(DEBUG) << "Run custom op bprop start";
    py::object grads_obj = hook_fn_(*converted_args);
    auto grads = check_bprop_out(grads_obj, py_args, bprop_cls_name_);
    MS_LOG(DEBUG) << "Run custom op bprop end";
    return std::make_shared<PyObjectRef>(grads);
  } catch (std::exception &bt) {
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunCellHookFunction(const py::tuple &py_args) const {
  const auto args_size = py_args.size();
  // Get the din passed to current bprop cut op.
  py::object grad_output = py_args[args_size - 1];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  if (!py::isinstance<py::tuple>(grad_output)) {
    grad_output = py::make_tuple(grad_output);
  }
  try {
    MS_LOG(DEBUG) << "Get cell dout " << ConvertPyObjToString(grad_output);
    if (hook_type_ == HookType::kBackwardPreHook) {
      MS_LOG(DEBUG) << "Run cell backward pre hook function start.";
      py::object ret = hook_fn_(grad_output);
      if (!py::isinstance<py::none>(ret)) {
        MS_LOG(DEBUG) << "Get cell backward pre hook new grad output " << ConvertPyObjToString(ret);
        const auto &code_obj = py::getattr(hook_fn_, "__code__");
        py::object co_name = py::getattr(code_obj, "co_name");
        CheckHookConsistency(UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
        grad_output = ret;
      }
      MS_LOG(DEBUG) << "Run cell backward pre hook function end.";
      return std::make_shared<PyObjectRef>(grad_output);
    }
    if (hook_type_ == HookType::kBackwardHook) {
      MS_LOG(DEBUG) << "Run cell backward hook function start.";
      py::object ret = hook_fn_(grad_output);
      if (py::isinstance<py::str>(ret)) {
        MS_LOG(DEBUG) << "Run cell " << ret.cast<std::string>() << " backward hook function the first time";
        unpair_backward_hook_grad_.emplace(ret.cast<std::string>(), hook_fn_);
        return std::make_shared<PyObjectRef>(grad_output);
      }
      if (py::isinstance<py::none>(ret)) {
        MS_LOG(DEBUG) << "Run cell backward hook function the second time with return None.";
      } else {
        MS_LOG(DEBUG) << "Get cell backward hook new grad input " << ConvertPyObjToString(ret);
        const auto &code_obj = py::getattr(hook_fn_, "__code__");
        py::object co_name = py::getattr(code_obj, "co_name");
        CheckHookConsistency(UnpackRetValueOfCellHook(ret), py_args[args_size - 1], co_name);
        grad_output = ret;
      }
      unpair_backward_hook_grad_.erase(GetValue<std::string>(this->GetAttr(kCellIDAttrName)));
      MS_LOG(DEBUG) << "Run cell backward hook function end.";
      return std::make_shared<PyObjectRef>(grad_output);
    }
    MS_LOG(EXCEPTION) << "Get unsupported hook function type";
  } catch (std::exception &bt) {
    unpair_backward_hook_grad_.clear();
    auto inst = pynative::PyNativeExecutor::GetInstance();
    inst->ClearRes();
    std::rethrow_exception(std::current_exception());
  }
}

BaseRef PrimitivePy::RunVariableHookFunction(const py::tuple &py_args) const {
  constexpr size_t grad_output_index = 0;
  if (py_args.size() != kSizeOne) {
    MS_LOG(EXCEPTION) << "Bprop cut run must in the following format: dout";
  }
  py::object grad_output = py_args[grad_output_index];
  grad_output = ConvertCTensorToPyTensor(grad_output);
  MS_LOG(DEBUG) << "Get grad output " << ConvertPyObjToString(grad_output);
  try {
    if (hook_type_ == HookType::kUnknown) {
      MS_LOG(EXCEPTION) << "Get unsupported hook type Unknown";
    }
    if (hook_type_ == HookType::kTensorHook) {
      MS_LOG(DEBUG) << "Run tensor hook function begin";
    } else {
      // Op maybe have multi outputs, so wrap to tuple for unitary.
      // Tensor hook just work on tensor, so keep origin input style
      if (!py::isinstance<py::tuple>(grad_output)) {
        grad_output = py::make_tuple(grad_output);
      }
      MS_LOG(DEBUG) << "Run HookBackward op function begin";
    }
    auto ret = hook_fn_(grad_output);
    if (!py::isinstance<py::none>(ret)) {
      MS_LOG(DEBUG) << "Get hook output " << ConvertPyObjToString(ret);
      grad_output = ret;
    }

    if (hook_type_ != HookType::kTensorHook) {
      const auto &code_obj = py::getattr(hook_fn_, "__code__");
      py::object co_name = py::getattr(code_obj, "co_name");
      CheckHookConsistency(UnpackRetValueOfCellHook(grad_output), py_args[grad_output_index], co_name);
    }
    MS_LOG(DEBUG) << (hook_type_ == HookType::kTensorHook ? "Run tensor hook function end"
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

BaseRef PrimitivePy::RunHookFunction(const VectorRef &args) const {
  py::tuple py_args = ConvertDatatoPyTuple(args);
  MS_LOG(DEBUG) << "Get input args size " << py_args.size() << ", args are " << ConvertPyObjToString(py_args);
  // For cell has custom bprop function
  if (hook_type_ == HookType::kCellCustomBprop) {
    MS_LOG(DEBUG) << "Run cell custom bprop";
    return RunCellCustomBpropFunction(py_args);
  }

  // For cell register hook
  if (hook_type_ == HookType::kBackwardPreHook || hook_type_ == HookType::kBackwardHook) {
    MS_LOG(DEBUG) << "Run cell backward hook";
    return RunCellHookFunction(py_args);
  }

  // For custom op, which define custrcut and bprop
  if (hook_type_ == HookType::kCustomOpBprop) {
    MS_LOG(DEBUG) << "Run custom op";
    return RunCustomOpBpropFunction(py_args);
  }

  // For hook use, include hook op and tensor register hook
  return RunVariableHookFunction(py_args);
}

py::function PrimitivePy::GetComputeFunction() const {
  static const char *const compute_func_name = "vm_impl";

  if (py::hasattr(python_obj_, compute_func_name)) {
    MS_LOG(DEBUG) << name() << " compute_func_name";
    py::function fn = python_obj_.attr(compute_func_name).cast<py::function>();
    return fn;
  }

  static const std::string vm_module = "mindspore.ops.vm_impl_registry";
  static const std::string get_vm_impl_fn = "get_vm_impl_fn";
  MS_LOG(DEBUG) << name() << ": get_vm_impl_fn";
  py::function get_fn = python_adapter::GetPyFn(vm_module, get_vm_impl_fn);
  py::function vm_fn = get_fn(python_obj_);
  if (py::isinstance<py::none>(vm_fn)) {
    vm_fn = get_fn(name());
  }
  if (py::isinstance<py::none>(vm_fn)) {
    MS_LOG(DEBUG) << "Cannot find " << python_obj_.attr("__class__").attr("__name__").cast<std::string>();
    vm_fn = mindspore::GetComputeFunction(Primitive::name());
  }
  return vm_fn;
}

py::dict PrimitivePy::GetAttrDict() {
  py::dict attr_dict;
  for (const auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

std::string PrimitivePy::HookTypeToString() const { return hook_type_with_str[hook_type_]; }

void PrimitivePy::CopyHookFunction(const PrimitivePyPtr &primitive_py) {
  MS_EXCEPTION_IF_NULL(primitive_py);
  SetHookFn(primitive_py->hook_fn(), primitive_py->hook_type());
  if (hook_type_ == HookType::kCellCustomBprop) {
    set_bprop_cls_name(primitive_py->bprop_cls_name_);
  }
}

BaseRef PrimitivePy::RunComputeFunction(const VectorRef &args) const {
  auto py_args = ConvertDatatoPyTuple(args);
  auto result = this->RunPyComputeFunction(py_args);
  if (py::isinstance<py::none>(result)) {
    return std::make_shared<BaseRef>(nullptr);
  }
  return std::make_shared<PyObjectRef>(result);
}

py::object PrimitivePy::RunPyComputeFunction(const py::tuple &py_args) const {
  auto func = this->GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    return py::none();
  }
  auto result = func(*py_args);
  return result;
}

bool PrimitivePy::HasComputeFunction() const {
  auto func = GetComputeFunction();
  return !py::isinstance<py::none>(func);
}

PrimitivePtr PrimitivePy::Clone() {
  auto clone_fn = python_obj_.attr("_clone");
  py::object obj_adapter = clone_fn();
  auto prim_adapter = obj_adapter.cast<PrimitivePyAdapterPtr>();
  auto prim = std::make_shared<PrimitivePy>(obj_adapter);
  prim_adapter->set_attached_primitive(prim);
  return prim;
}

py::dict PrimitivePy::RunInfer(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER;
  }
  auto infer_fuc = python_obj_.attr(PY_PRIM_METHOD_INFER);
  return infer_fuc(*args);
}

void PrimitivePy::RunCheck(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_CHECK)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_CHECK;
  }
  auto check_func = python_obj_.attr(PY_PRIM_METHOD_CHECK);
  (void)check_func(*args);
}

py::object PrimitivePy::RunInferValue(const py::tuple &args) {
  if (!HasPyObj()) {
    MS_LOG(EXCEPTION) << "[" << this->ToString() << "]: pyobj is empty";
  }
  // Python obj could be replaced as None, so it will losed the original info when throw exception in python.
  if (!py::hasattr(python_obj_, PY_PRIM_METHOD_INFER_VALUE)) {
    MS_LOG(EXCEPTION) << "prim:" << ToString() << " has no attr:" << PY_PRIM_METHOD_INFER_VALUE;
  }
  auto infer_value = python_obj_.attr(PY_PRIM_METHOD_INFER_VALUE);
  return infer_value(*args);
}

void PrimitivePy::ProcessUnPairedCellHook(bool execute_hook_fn) {
  if (execute_hook_fn) {
    for (const auto &[cell_id, hook_fn] : unpair_backward_hook_grad_) {
      MS_LOG(DEBUG) << "Run unpair backward cell hook " << cell_id;
      (void)hook_fn(py::make_tuple(py::none()));
    }
  }
  unpair_backward_hook_grad_.clear();
}

void PrimitivePy::ClearHookRes() { unpair_backward_hook_grad_.clear(); }

PrimitivePyAdapter::PrimitivePyAdapter(const py::str &name) : id_(MakeId()), name_(name) {}

PrimitivePyAdapter::PrimitivePyAdapter(const PrimitivePyAdapter &adapter)
    : const_prim_(adapter.const_prim_),
      inplace_prim_(adapter.inplace_prim_),
      id_(adapter.id_),
      name_(adapter.name_),
      instance_name_(adapter.instance_name_),
      prim_type_(adapter.prim_type_),
      attrs_(adapter.attrs_),
      const_input_indexes_(adapter.const_input_indexes_),
      signatures_(adapter.signatures_),
      hook_fn_(adapter.hook_fn_) {}

PrimitivePyAdapter &PrimitivePyAdapter::operator=(const PrimitivePyAdapter &other) {
  if (this == &other) {
    return *this;
  }
  const_prim_ = other.const_prim_;
  inplace_prim_ = other.inplace_prim_;
  id_ = other.id_;
  name_ = other.name_;
  instance_name_ = other.instance_name_;
  prim_type_ = other.prim_type_;
  attrs_ = other.attrs_;
  const_input_indexes_ = other.const_input_indexes_;
  signatures_ = other.signatures_;
  hook_fn_ = other.hook_fn_;
  return *this;
}

void PrimitivePyAdapter::AddPyAttr(const py::str &name, const py::object &obj) {
  std::string attr_name = name;
  ValuePtr converted_res = nullptr;
  if (py::isinstance<py::module>(obj)) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " not support py::module to be attribute value; primitive name: " << this->name_
                      << ", attribute name: " << attr_name << " attribute value: " << py::str(obj);
  }
  bool converted = parse::ConvertData(obj, &converted_res);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                      << " convert python obj to MindSpore obj failed; primitive name: " << this->name_
                      << ", attribute name:" << attr_name << ", attribute value:" << py::str(obj)
                      << ", attribute type:" << py::cast<std::string>(obj.attr("__class__").attr("__name__"));
  }
  if (kOpAttrNameReplaceMap.find(attr_name) != kOpAttrNameReplaceMap.end()) {
    attr_name = kOpAttrNameReplaceMap[attr_name];
  }
  (void)CheckAndConvertUtils::ConvertAttrValueToInt(this->name_, name, &converted_res);
  if (attr_name == "primitive_target") {
    MS_EXCEPTION_IF_NULL(converted_res);
    if (!converted_res->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
    auto target = GetValue<std::string>(converted_res);
    if (!target.empty() && target != kCPUDevice && target != kGPUDevice && target != kAscendDevice &&
        target != "Device") {
      MS_LOG(EXCEPTION) << "Call 'add_attr' to add attribute to primitive '" << this->name_
                        << "' failed, value of attribute 'primitive_target' must be CPU|GPU|Ascend but got "
                        << py::str(obj);
    }
  }

  // If it's func graph, to reserve all used func graphs.
  if (converted_res->isa<FuncGraph>()) {
    const auto &fg = dyn_cast<FuncGraph>(converted_res);
    MS_EXCEPTION_IF_NULL(fg);
    fg->set_reserved(true);
    auto manager = Manage({fg}, false);
    const auto &total_used_fg = manager->func_graphs_used_total(fg);
    for (const auto &used_fg : total_used_fg) {
      used_fg->set_reserved(true);
    }
  }

  attrs_[attr_name] = converted_res;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->AddAttr(attr_name, converted_res);
  }
}

void PrimitivePyAdapter::DelPyAttr(const py::str &name) {
  (void)attrs_.erase(name);
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    (void)prim->DelAttr(name);
  }
}

py::dict PrimitivePyAdapter::GetAttrDict() {
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    return prim->GetAttrDict();
  }

  py::dict attr_dict;
  for (const auto &attr : attrs_) {
    attr_dict[py::str(attr.first)] = ValueToPyData(attr.second);
  }
  return attr_dict;
}

void PrimitivePyAdapter::set_prim_type(const PrimType t) {
  prim_type_ = t;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_prim_type(t);
  }
}

void PrimitivePyAdapter::set_const_prim(bool is_const_prim) {
  const_prim_ = is_const_prim;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_prim(is_const_prim);
  }
}

void PrimitivePyAdapter::set_inplace_prim(bool is_inplace_prim) {
  inplace_prim_ = is_inplace_prim;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_inplace_prim(is_inplace_prim);
  }
}

void PrimitivePyAdapter::set_const_input_indexes(const std::vector<size_t> &const_input_indexes) {
  const_input_indexes_ = const_input_indexes;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_const_input_indexes(const_input_indexes);
  }
}

void PrimitivePyAdapter::set_signatures(const std::vector<Signature> &signatures) {
  signatures_ = signatures;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_signatures(signatures);
  }
}

void PrimitivePyAdapter::SetHookFn(const py::function &hook_fn, HookType hook_type) {
  hook_fn_ = hook_fn;
  hook_type_ = hook_type;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->SetHookFn(hook_fn, hook_type_);
  }
}

void PrimitivePyAdapter::set_instance_name(const std::string &s) {
  instance_name_ = s;
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_instance_name(s);
  }
}

void PrimitivePyAdapter::set_attached_primitive(const PrimitivePyPtr &prim) {
  if (attached_primitive_.lock() != nullptr) {
    MS_LOG(EXCEPTION) << "PrimitivePyAdapter can't attach to multi Primitive.";
  }
  MS_EXCEPTION_IF_NULL(prim);
  attached_primitive_ = prim;
}

void PrimitivePyAdapter::SetUserData(const py::str &key, const py::object &value) {
  const std::string name = std::string("__primitive_user_data_") + key.cast<std::string>();
  const auto &primitive_data = std::make_shared<PrimitiveUserData>();
  primitive_data->obj = value;
  // Set into primitive adapter.
  set_user_data<PrimitiveUserData>(name, primitive_data);
  // Set in primitive.
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    prim->set_user_data<PrimitiveUserData>(name, primitive_data);
  }
}

py::object PrimitivePyAdapter::GetUserData(const py::str &key) const {
  const std::string name = std::string("__primitive_user_data_") + key.cast<std::string>();
  // Get from primitive.
  auto prim = attached_primitive_.lock();
  if (prim != nullptr) {
    const auto primitive_data = prim->user_data<PrimitiveUserData>(name);
    return primitive_data->obj;
  }
  // Get from primtive adapter.
  const auto primitive_data = user_data<PrimitiveUserData>(name);
  return primitive_data->obj;
}

void PrimitiveFunctionAdapter::set_label(const std::string &label, const py::object &value) {
  ValuePtr converted_value = nullptr;
  if (!parse::ConvertData(value, &converted_value)) {
    MS_LOG(INTERNAL_EXCEPTION) << "For '" << PrimitiveFunctionAdapter::name() << "', Convert data failed.";
  }
  attached_primitive_function_->AddAttr(label, converted_value);
}

py::object PrimitiveFunctionAdapter::clone() {
  const auto op_path = "mindspore.ops.primitive";
  const auto func = "_create_primitive_function_obj";
  py::object prim_func_adapter_obj = python_adapter::CallPyFn(op_path, func);
  prim_func_adapter_obj.cast<PrimitiveFunctionAdapterPtr>()->set_attached_primitive_function(
    attached_primitive_function_->Clone());
  return prim_func_adapter_obj;
}

void RegPrimitive(const py::module *m) {
  (void)py::enum_<HookType>(*m, "HookType", py::arithmetic())
    .value("CustomOpBprop", HookType::kCustomOpBprop)
    .value("CellCustomBprop", HookType::kCellCustomBprop)
    .value("HookBackward", HookType::kHookBackwardOp)
    .value("TensorHook", HookType::kTensorHook)
    .value("BackwardPreHook", HookType::kBackwardPreHook)
    .value("BackwardHook", HookType::kBackwardHook);
  (void)py::enum_<PrimType>(*m, "prim_type", py::arithmetic())
    .value("unknown", PrimType::kPrimTypeUnknown)
    .value("builtin", PrimType::kPrimTypeBuiltIn)
    .value("py_infer_shape", PrimType::kPrimTypePyInfer)
    .value("user_custom", PrimType::kPrimTypeUserCustom)
    .value("py_infer_check", PrimType::kPrimTypePyCheck);
  (void)py::class_<PrimitivePyAdapter, std::shared_ptr<PrimitivePyAdapter>>(*m, "Primitive_")
    .def_readonly(PYTHON_PRIMITIVE_FLAG, &PrimitivePyAdapter::parse_info_)
    .def(py::init<py::str &>())
    .def("add_attr", &PrimitivePyAdapter::AddPyAttr, "add primitive attr")
    .def("del_attr", &PrimitivePyAdapter::DelPyAttr, "del primitive attr")
    .def("get_attr_dict", &PrimitivePyAdapter::GetAttrDict, "get primitive attr")
    .def("set_prim_type", &PrimitivePyAdapter::set_prim_type, "Set primitive type.")
    .def("set_const_prim", &PrimitivePyAdapter::set_const_prim, "Set primitive is const.")
    .def("set_inplace_prim", &PrimitivePyAdapter::set_inplace_prim, "Set primitive is inplace primitive.")
    .def("set_const_input_indexes", &PrimitivePyAdapter::set_const_input_indexes, "Set primitive const input indexes.")
    .def("set_signatures", &PrimitivePyAdapter::set_signatures, "Set primitive inputs signature.")
    .def("set_hook_fn", &PrimitivePyAdapter::SetHookFn, "Add primitive hook function.")
    .def("set_instance_name", &PrimitivePyAdapter::set_instance_name, "Set primitive instance name.")
    .def("set_user_data", &PrimitivePyAdapter::SetUserData, "Set primitive user data.")
    .def("get_user_data", &PrimitivePyAdapter::GetUserData, "Get primitive user data.");
}

void RegPrimitiveFunction(const py::module *m) {
  (void)py::class_<PrimitiveFunctionAdapter, std::shared_ptr<PrimitiveFunctionAdapter>>(*m, "PrimitiveFunction_")
    .def_readonly(PYTHON_PRIMITIVE_FUNCTION_FLAG, &PrimitiveFunctionAdapter::parse_info_)
    .def(py::init<>())
    .def_property_readonly("name", &PrimitiveFunctionAdapter::name, "Get function name.")
    .def("has_label", &PrimitiveFunctionAdapter::has_label, "Has function attr.")
    .def("set_label", &PrimitiveFunctionAdapter::set_label, "Set function attr.")
    .def("get_label", &PrimitiveFunctionAdapter::get_label, "Get function attr.")
    .def("clone", &PrimitiveFunctionAdapter::clone, "Clone a Primitive and create a PrimitiveFunctionAdapter.");
}
}  // namespace mindspore
