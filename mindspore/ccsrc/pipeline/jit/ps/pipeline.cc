/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/pipeline.h"

#include <memory>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <functional>
#include "mindspore/ops/op_def/framework_ops.h"
#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"
#include "ir/param_info.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/pipeline_jit.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "pipeline/jit/ps/compile_cache_manager.h"
#include "pipeline/pynative/pynative_execute.h"
#include "pipeline/pynative/op_function/converter.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/graph_util/flops_collection.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/allreduce_fusion/step_allreduce_fusion.h"
#include "frontend/parallel/pass/handle_group_info.h"
#include "frontend/parallel/step_assigned_parallel.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/expander/utils.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/python_utils.h"
#include "include/backend/debug/execute_order_tracker/execute_order_tracker.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "utils/info.h"
#include "utils/crypto.h"
#include "utils/phase.h"
#include "utils/compile_config.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/stub_tensor.h"
#include "utils/interpret_node_recorder.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pipeline/jit/ps/event_message_print.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/common.h"
#include "load_mindir/load_model.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/common/session/executor_manager.h"
#include "backend/common/session/session_factory.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/stream_synchronizer.h"
#include "include/common/fallback.h"
#include "include/common/profiler.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "include/backend/distributed/init.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/debug/tft_adapter/tft_wait_sem.h"
#include "kernel/graph_kernel/graph_kernel_builder_manager.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "symbolic_shape/symbol_info.h"
#include "include/common/symbol_engine/symbol_engine_impl.h"
#include "pipeline/jit/ps/load_mindir.h"
#include "load_mindir/infer_mindir.h"
#include "pipeline/jit/ps/pass_config.h"

#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/acl_dump_json_writer.h"
#include "abstract/abstract_value.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/constants.h"
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_cache/ps_data_prefetch.h"
#include "include/backend/distributed/cluster/cluster_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/distributed/embedding_cache/data_queue_manager.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "ir/cell.h"
#endif

#include "pybind_api/ir/log_adapter_py.h"  // Only include one-time in the whole project.
#include "pybind_api/ir/py_execute_py.h"   // Only include one-time in the whole project.
#include "pybind_api/ir/tensor_register/auto_generate/tensor_func_utils.h"
#include "include/common/utils/compile_cache_context.h"

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaSparseTensor = mindspore::tensor::MetaSparseTensor;
using CSRTensor = mindspore::tensor::CSRTensor;
using COOTensor = mindspore::tensor::COOTensor;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using DeviceTensor = mindspore::device::DeviceAddress;

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";
const char IR_TYPE_MINDIR[] = "mind_ir";

GraphExecutorPyPtr GraphExecutorPy::executor_ = nullptr;
std::mutex GraphExecutorPy::instance_lock_;

std::unordered_map<abstract::AbstractBasePtrList, uint64_t, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  kArgsCache;
std::unordered_map<PyObject *, abstract::AbstractBasePtrList> kCellArgsMap;

namespace {
constexpr size_t kPhaseSavePrefixLen = 5;

bool CheckAllTensor(const ValueTuplePtr &value_tuple) {
  auto elements = value_tuple->value();
  for (auto element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (!(element->isa<ValueTuple>() && CheckAllTensor(element->cast<ValueTuplePtr>())) &&
        !(element->isa<MetaTensor>())) {
      return false;
    }
  }
  return true;
}

bool Mutable(const py::object &obj, const ValuePtr &value) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value->isa<MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

bool CheckAndConvertToVariableLenSequence(const py::object &obj, AbstractBasePtr abs) {
  constexpr char variable_len_attr[] = "__ms_dynamic_len__";
  bool dynamic_len = (py::hasattr(obj, variable_len_attr) && py::cast<bool>(py::getattr(obj, variable_len_attr)));
  if (!dynamic_len) {
    return false;
  }
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For mutable, when the dynamic_len the True, the first input should be"
                            << " list or tuple, but got: " << abs->ToString();
  }
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  abs_seq->CheckAndConvertToDynamicLenSequence();
  return true;
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool EnableTupleBroaden(const ValuePtr &value, bool enable_tuple_broaden) {
  return enable_tuple_broaden && value->isa<ValueTuple>() && CheckAllTensor(value->cast<ValueTuplePtr>());
}

bool GradForScalar(const ValuePtr &value) {
  return (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1") &&
         value->isa<Scalar>();
}

bool CheckArgValid(const py::handle &arg) {
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto vector_arg = py::cast<py::list>(arg);
    return std::all_of(vector_arg.begin(), vector_arg.end(), CheckArgValid);
  }

  if (py::isinstance<py::dict>(arg)) {
    auto dict_arg = py::cast<py::dict>(arg);
    return std::all_of(dict_arg.begin(), dict_arg.end(), [](const auto &pair) { return CheckArgValid(pair.second); });
  }

  if (py::isinstance<Tensor>(arg) || IsStubTensor(arg)) {
    auto tensor = IsStubTensor(arg) ? ConvertStubTensor(arg) : py::cast<TensorPtr>(arg);
    if (tensor->data_type() == kNumberTypeBool) {
      MS_LOG(INFO) << "It is not recommended to use a tensor of bool data type as network input, which may cause "
                   << "operator compilation failure. For more details, please refer to the FAQ at "
                   << "https://mindspore.cn/search?[AddN]%20input(kNumberTypeBool.";
    }
  }

  return IsStubTensor(arg) || py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg) ||
         py::isinstance<py::none>(arg) || py::isinstance<Number>(arg) || py::isinstance<py::str>(arg) ||
         py::isinstance<Tensor>(arg) || py::isinstance<CSRTensor>(arg) || py::isinstance<COOTensor>(arg);
}

std::string GetCompileExceptionInfo() {
  std::ostringstream oss;
  trace::GetTraceStackInfo(oss);
  return oss.str();
}

std::map<string, string> GenerateJitConfigMap(const py::dict &jit_config) {
  std::map<string, string> ret{};
  for (auto jit_param = jit_config.begin(); jit_param != jit_config.end(); ++jit_param) {
    auto param_name = py::cast<std::string>(jit_param->first);
    auto param_value = py::cast<std::string>(jit_param->second);
    ret[param_name] = param_value;
  }
  return ret;
}

void RecordInitStatus() {
  static bool printed = false;
  if (!printed) {
    MS_LOG(INFO) << "Status record: system init.";
    printed = true;
  }
}

void RecordExitStatus() { MS_LOG(INFO) << "Status record: system exit."; }

std::string ToOrdinal(const size_t &i) {
  auto suffix = "th";
  if (i == kIndex1) {
    suffix = "st";
  } else if (i == kIndex2) {
    suffix = "nd";
  } else if (i == kIndex3) {
    suffix = "rd";
  }
  return std::to_string(i) + suffix;
}

kernel::PyExecuteOutputUserDataPtr GetUserDataFromAddress(const py::object &res) {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return nullptr;
  }

  if (py::isinstance<tensor::Tensor>(res) || IsStubTensor(res)) {
    auto res_tensor = IsStubTensor(res) ? ConvertStubTensor(res) : res.cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(res_tensor);
    if (res_tensor->device_address() != nullptr) {
      auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(res_tensor->device_address());
      MS_LOG(DEBUG) << "res tensor_address:" << tensor_address;
      MS_EXCEPTION_IF_NULL(tensor_address);
      if (tensor_address->user_data() != nullptr) {
        return tensor_address->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
      }
    }
  }
  return nullptr;
}

template <typename T>
py::object GetVectorRefPyDataWithAbstract(const VectorRef &value_list, const abstract::AbstractSequencePtr &seq_abs) {
  auto value_size = value_list.size();
  auto ret = T(value_size);

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  size_t ref_idx = 0;
  for (size_t i = 0; i < seq_abs->size(); ++i) {
    auto elem_abs = seq_abs->elements()[i];
    if (elem_abs->isa<abstract::AbstractNone>() && !allow_fallback_runtime) {
      continue;
    }
    ret[ref_idx] = BaseRefToPyDataWithUserData(value_list[ref_idx], elem_abs);
    ref_idx++;
  }
  if (ref_idx != value_size) {
    MS_LOG(EXCEPTION) << "The size of elements (excluding None) should be equal to " << value_size << ", but got "
                      << ref_idx;
  }
  return ret;
}

py::object GetVectorRefPyData(const VectorRef &value_list, const AbstractBasePtr &abs) {
  if (abs == nullptr || abs->isa<abstract::AbstractCSRTensor>() || abs->isa<abstract::AbstractCOOTensor>() ||
      abs->isa<abstract::AbstractAny>()) {
    return BaseRefToPyData(value_list, abs);
  }
  // Need to consider AbstractAny with vector ref scene later.
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Can not convert vector ref with abstract " << abs->ToString();
  }
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    return BaseRefToPyData(value_list, abs);
  }
  if (seq_abs->isa<abstract::AbstractTuple>()) {
    return GetVectorRefPyDataWithAbstract<py::tuple>(value_list, seq_abs);
  }
  return GetVectorRefPyDataWithAbstract<py::list>(value_list, seq_abs);
}

void AddManager(const FuncGraphManagerPtr &manager, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<FuncGraph>()) {
    auto fg = value->cast<FuncGraphPtr>();
    manager->AddFuncGraph(fg);
  }
  if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &elem : value_sequence->value()) {
      AddManager(manager, elem);
    }
  }
  if (value->isa<ValueDictionary>()) {
    for (const auto &elem : value->cast<ValueDictionaryPtr>()->value()) {
      AddManager(manager, elem.second);
    }
  }
}

void CheckShapeConsistency(const abstract::ShapePtr &compile_shape, const abstract::ShapePtr &args_shape,
                           const std::string &target_str, size_t index) {
  MS_EXCEPTION_IF_NULL(compile_shape);
  MS_EXCEPTION_IF_NULL(args_shape);
  if (*compile_shape == *args_shape) {
    return;
  }

  auto compile_shape_vec = compile_shape->shape();
  auto args_shape_vec = args_shape->shape();

  if (!IsDynamicRank(compile_shape_vec)) {
    if (!args_shape_vec.empty() && compile_shape_vec.size() != args_shape_vec.size()) {
      MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str << ", the dims of "
                               << index + 1 << "th input must be the same as expected, "
                               << "but got expected: " << compile_shape_vec.size()
                               << ", and input: " << args_shape_vec.size() << "!";
    }

    for (size_t i = 0; i < compile_shape_vec.size(); ++i) {
      if (compile_shape_vec[i] == abstract::Shape::kShapeDimAny || compile_shape_vec[i] == args_shape_vec[i]) {
        continue;
      }
      MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str << ", the shape of "
                               << index + 1 << "th input must be the same as expected, "
                               << "but got expected: " << compile_shape_vec[i] << ", and input: " << args_shape_vec[i]
                               << "!";
    }
  }
}

inline void CheckSizeConsistency(const AbstractBasePtrList &compile_abstracts,
                                 const AbstractBasePtrList &args_abstracts, const std::string &target_str,
                                 bool dynamic_len = false) {
  if (!dynamic_len && compile_abstracts.size() != args_abstracts.size()) {
    MS_EXCEPTION(ValueError) << "For " << target_str << " and tuple(list) in " << target_str
                             << ", the length of input must be equal to expected one, but got expected: "
                             << compile_abstracts.size() << " and input: " << args_abstracts.size() << "!";
  }
  if (dynamic_len && compile_abstracts.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << target_str << ", the dynamic_len compile arguments should not be empty!";
  }
}

void CheckAbstractConsistency(const AbstractBasePtrList &compile_abstracts, const AbstractBasePtrList &args_abstracts,
                              const std::string &target_str, bool dynamic_len = false) {
  CheckSizeConsistency(compile_abstracts, args_abstracts, target_str, dynamic_len);
  for (size_t i = 0; i < args_abstracts.size(); ++i) {
    auto compile_abs = dynamic_len ? compile_abstracts[0] : compile_abstracts[i];
    auto args_abs = args_abstracts[i];
    auto is_compile_var = compile_abs->BuildValue()->ContainsValueAny();
    auto is_args_var = args_abs->BuildValue()->ContainsValueAny();
    if (is_compile_var != is_args_var) {
      MS_EXCEPTION(TypeError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                              << "th should be " << (is_compile_var ? "mutable" : "static") << " one, but got "
                              << (is_args_var ? "mutable" : "static") << "!";
    }

    if (is_compile_var) {
      if (compile_abs->isa<abstract::AbstractTensor>() && args_abs->isa<abstract::AbstractTensor>()) {
        auto compile_tensor = compile_abs->cast<abstract::AbstractTensorPtr>();
        auto args_tensor = args_abs->cast<abstract::AbstractTensorPtr>();

        // Check shape's consistency.
        auto compile_shape = compile_tensor->shape();
        auto args_shape = args_tensor->shape();
        CheckShapeConsistency(compile_shape, args_shape, target_str, i);

        auto compile_element = compile_tensor->element();
        auto args_element = args_tensor->element();
        if (!common::IsEqual(compile_element, args_element)) {
          MS_EXCEPTION(TypeError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                                  << "th type should be " << compile_tensor->BuildType()->ToString() << ", but got "
                                  << args_tensor->BuildType()->ToString() << "!";
        }
      } else if (compile_abs->isa<abstract::AbstractSequence>() && args_abs->isa<abstract::AbstractSequence>()) {
        auto compile_sequence = compile_abs->cast<abstract::AbstractSequencePtr>();
        auto args_sequence = args_abs->cast<abstract::AbstractSequencePtr>();
        CheckAbstractConsistency(compile_sequence->elements(), args_sequence->elements(), target_str,
                                 compile_sequence->dynamic_len());
      } else {
        if (!common::IsEqual(compile_abs, args_abs)) {
          MS_EXCEPTION(ValueError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                                   << "th should be" << compile_abs->ToString() << ", but got " << args_abs->ToString()
                                   << "!";
        }
      }
    } else if (compile_abs->isa<abstract::AbstractList>() && args_abs->isa<abstract::AbstractList>()) {
      auto compile_sequence = compile_abs->cast<abstract::AbstractSequencePtr>();
      auto args_sequence = args_abs->cast<abstract::AbstractSequencePtr>();
      CheckAbstractConsistency(compile_sequence->elements(), args_sequence->elements(), target_str);
    } else {
      if (!common::IsEqual(compile_abs, args_abs)) {
        MS_EXCEPTION(ValueError) << "For " << target_str << " or tuple(list) in " << target_str << ", the " << i + 1
                                 << "th should be" << compile_abs->ToString() << ", but got " << args_abs->ToString()
                                 << "!";
      }
    }
  }
}

inline pid_t GetCurrentPID() {
#if defined(_WIN32) || defined(_WIN64)
  return GetCurrentProcessId();
#else
  return getpid();
#endif
}

py::object GetSelfFromArgs(const py::object &args) {
  if (!py::isinstance<py::tuple>(args)) {
    return py::object();
  }
  auto args_tuple = py::cast<py::tuple>(args);
  if (args_tuple.size() == 0) {
    return py::object();
  }
  py::object first_arg = args_tuple[0];
  if (!py::isinstance<Cell>(first_arg)) {
    return py::object();
  }
  return first_arg;
}
}  // namespace

void AddManagerForFuncGraphArgs(const ResourcePtr &resource, const ValuePtrList &arguments) {
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &arg : arguments) {
    AddManager(manager, arg);
  }
}

AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden) {
  bool broaden = TensorArgMutable(arg, value) || Mutable(arg, value) || value->isa<MetaSparseTensor>() ||
                 EnableTupleBroaden(value, enable_tuple_broaden) || GradForScalar(value);
  auto ret = abstract::ToAbstract(value, nullptr, nullptr);
  if (broaden) {
    ret = AbstractBroaden(ret);
  }
  auto is_dynamic_len = CheckAndConvertToVariableLenSequence(arg, ret);
  if (fallback::EnableFallbackListDictInplace() && !broaden && !is_dynamic_len) {
    // Attach corresponding list python object for constant list input.
    fallback::AttachPyObjToAbs(ret, arg, false);
  }
  return ret;
}

void SetLoopCount(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  if (func_graph != nullptr && func_graph->manager() != nullptr) {
    auto manager = func_graph->manager();
    size_t graph_nums = manager->func_graphs().size();
    int64_t loop_size = ConfigManager::GetInstance().iter_num();
    const auto context_ptr = MsContext::GetInstance();
    bool enable_mind_rt = context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT);
    if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
      resource->set_vm_loop(!(context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK) || enable_mind_rt), loop_size);
    } else if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
      bool run_with_mind_rt = graph_nums == 1 || enable_mind_rt;
      resource->set_vm_loop(!run_with_mind_rt, loop_size);
    }
    MS_LOG(INFO) << "Change vm_loop_flag to " << resource->vm_loop_flag() << ", set loop_size to " << loop_size;
  }
}

void ResetId(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto need_dump = common::GetCompileConfig("DUMP_VALIDATE_BEFORE_RESET_ID");
  if (context->CanDump(kIntroductory) && need_dump == "1") {
    FuncGraphPtr graph = resource->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    DumpIR("validate_before_reset_id.ir", graph, true, kWholeStack);
  }
#endif
  mindspore::id_generator::reset_id();
  const auto &all_nodes = TopoSort(resource->func_graph()->get_return(), SuccDeeperSimple);
  auto ge_mode = common::AnfAlgo::IsBackendGe();
  for (const auto &node : all_nodes) {
    if (node != nullptr && node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->set_fullname_with_scope("");
      if (!ge_mode) {
        (void)cnode->fullname_with_scope();
      }
    }
  }
}

std::string GetObjDesc(const py::object &source) {
  std::string obj_desc;
  if (py::hasattr(source, parse::PYTHON_PARSE_METHOD)) {
    auto cell_class_name = source.attr("__class__").attr("__name__");
    auto jit_name = source.attr(parse::PYTHON_PARSE_METHOD);
    obj_desc = "'" + py::cast<std::string>(cell_class_name) + "." + py::cast<std::string>(jit_name) + "'";
  } else {
    if (py::hasattr(source, "__name__")) {
      auto jit_name = source.attr("__name__");
      obj_desc = "'" + py::cast<std::string>(jit_name) + "'";
    } else if (py::isinstance<Cell>(source)) {
      auto cell_class_name = source.attr("__class__").attr("__name__");
      obj_desc = "'" + py::cast<std::string>(cell_class_name) + ".construct'";
    } else {
      MS_EXCEPTION(TypeError) << "The source object is invalid: " << py::str(source);
    }
  }
  return obj_desc;
}

void CheckArgsValid(const py::object &source, const py::tuple &args) {
  if (!IS_OUTPUT_ON(mindspore::kInfo)) {
    return;
  }
  for (size_t i = 0; i < args.size(); i++) {
    if (!CheckArgValid(args[i])) {
      MS_LOG(INFO) << "The " << ToOrdinal(i + 1) << " arg type is " << args[i].get_type() << ", value is '"
                   << py::str(args[i]) << "'.";
    }
  }
}

void ExecutorPy::CheckArgumentsConsistency(const py::tuple &compile_args, const py::tuple &args_list,
                                           const py::object &target) {
  if ((!py::isinstance<py::str>(target))) {
    MS_EXCEPTION(TypeError) << "The `target` must be string!";
  }
  std::string target_str = py::cast<std::string>(target);
  if (compile_args.size() != args_list.size()) {
    MS_EXCEPTION(ValueError) << "For " << target_str
                             << ", the length of input must be equal to expected one, but got expected: "
                             << compile_args.size() << " and input: " << args_list.size() << "!";
  }

  AbstractBasePtrList compile_abstracts;
  compile_abstracts.reserve(compile_args.size());
  AbstractBasePtrList args_abstracts;
  args_abstracts.reserve(compile_args.size());
  for (size_t i = 0; i < compile_args.size(); ++i) {
    ValuePtr compile_args_converted = nullptr;
    if (!parse::ConvertData(compile_args[i], &compile_args_converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th compiling argument failed, the argument type is "
                                 << compile_args[i].get_type() << ", value is '" << py::str(compile_args[i]) << "'.";
    }
    compile_abstracts.push_back(ArgsToAbstract(compile_args[i], compile_args_converted));

    ValuePtr args_converted = nullptr;
    if (!parse::ConvertData(args_list[i], &args_converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th input argument failed, the argument type is "
                                 << args_list[i].get_type() << ", value is '" << py::str(args_list[i]) << "'.";
    }
    args_abstracts.push_back(ArgsToAbstract(args_list[i], args_converted));
  }

  CheckAbstractConsistency(compile_abstracts, args_abstracts, target_str, false);
}

py::object ExecutorPy::GenerateArgumentsKey(const py::object &obj, const py::tuple &args, const py::dict &kwargs,
                                            bool enable_tuple_broaden) {
  MS_LOG(DEBUG) << "GenerateArgumentsKey, args size: " << args.size()
                << ", enable_tuple_broaden: " << enable_tuple_broaden;
  abstract::AbstractBasePtrList args_abs;
  ClearCurConvertInput();
  for (std::size_t i = 0; i < args.size(); i++) {
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(args[i], &converted)) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for " << i << "th argument failed, the argument type is "
                                 << args[i].get_type() << ", value is '" << py::str(args[i]) << "'.";
    }
    AbstractBasePtr abs = ArgsToAbstract(args[i], converted, enable_tuple_broaden);

    (void)args_abs.emplace_back(abs);
    // The 'converted' maybe a Parameter, we need connect it to the Parameter of func graph,
    // so we keep all inputs for subsequent procedure.
    (void)cur_convert_input_.emplace(args[i].ptr(), std::make_pair(converted, abs));
  }
  for (const auto &item : kwargs) {
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.first), &key) &&
                   parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "ConvertData for argument (" << py::str(item.first) << ": " << py::str(item.second)
                                 << ") failed.";
    }
    AbstractBasePtr value_abs = ArgsToAbstract(py::cast<py::object>(item.second), value, enable_tuple_broaden);
    auto keyword_arg_abs = std::make_shared<abstract::AbstractKeywordArg>(GetValue<std::string>(key), value_abs);

    (void)args_abs.emplace_back(keyword_arg_abs);
    (void)cur_convert_input_.emplace(item.first.ptr(), std::make_pair(value, keyword_arg_abs));
  }

  // If cache matched no need CheckArgsValid
  auto iter = kArgsCache.find(args_abs);
  if (iter != kArgsCache.end()) {
    return py::int_(iter->second);
  }

  static uint64_t key_counter = 0;
  kArgsCache[args_abs] = key_counter;
  if (!py::isinstance<py::none>(obj)) {
    kCellArgsMap[obj.ptr()] = args_abs;
  }
  MS_LOG(INFO) << "Generate a new compile key for new args, key: " << key_counter;
  if (IS_OUTPUT_ON(mindspore::kInfo)) {
    std::ostringstream buffer;
    buffer << "New cached args:"
           << "\n";
    for (size_t i = 0; i < args_abs.size(); ++i) {
      buffer << "Arg[" << i << "]: " << args_abs[i]->ToString() << "\n";
    }
    MS_LOG(INFO) << buffer.str();
  }
  return py::int_(key_counter++);
}

void ExecutorPy::ClearCompileArgumentsResource() {
  // Clear global converted args saved in GenerateArgumentsKey.
  ClearCurConvertInput();
}

void ClearArgCache(const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return;
  }
  auto iter = kCellArgsMap.find(obj.ptr());
  if (iter != kCellArgsMap.end()) {
    (void)kArgsCache.erase(iter->second);
    (void)kCellArgsMap.erase(iter);
  }
}

void ExecutorPy::ClearCurConvertInput() { cur_convert_input_.clear(); }

void GraphExecutorPy::ParentBeforeFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy prepare before fork.";
  MS_LOG(DEBUG) << "Stop AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Stop();
  MS_LOG(DEBUG) << "GraphExecutorPy prepare before fork done.";
}

void GraphExecutorPy::ParentAfterFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy in parent process reinitialize after fork.";
  MS_LOG(DEBUG) << "Restart AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Start();
  MS_LOG(DEBUG) << "GraphExecutorPy in parent process reinitialize after fork done.";
}

void GraphExecutorPy::ChildAfterFork() {
  MS_LOG(DEBUG) << "GraphExecutorPy in child process reinitialize after fork.";
  MS_LOG(DEBUG) << "Restart AnalysisSchedule tasks.";
  abstract::AnalysisSchedule::GetInstance().Start();
  MS_LOG(DEBUG) << "GraphExecutorPy in child process reinitialize after fork done.";
}

py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    std::shared_ptr<Tensor> m_tensor = nullptr;
    bool is_tensor = false;
    if (py::isinstance<Tensor>(arg_obj)) {
      m_tensor = arg_obj.cast<std::shared_ptr<Tensor>>();
      is_tensor = true;
    } else if (IsStubTensor(arg_obj)) {
      m_tensor = ConvertStubTensor(arg_obj);
      is_tensor = true;
    }
    if (is_tensor && m_tensor == nullptr) {
      MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
      return false;
    }

    if (m_tensor != nullptr) {
      MS_LOG(DEBUG) << "Verify Tensor";
      auto sig = input_signature[count].cast<std::shared_ptr<MetaTensor>>();
      ShapeVector sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      ShapeVector tensor_shape = m_tensor->shape_c();
      if (tensor_shape != sig_shape) {
        MS_LOG(ERROR) << "Python input shape is incompatible with input_signature";
        return false;
      }

      if (*m_tensor->Dtype() != *sig_type) {
        MS_LOG(ERROR) << "Python input type(" << m_tensor->Dtype()->ToString() << ") incompatible with input_signature("
                      << sig_type->ToString() << ")";
        return false;
      }
    }
    count++;
  }

  return true;
}

ResourcePtr ExecutorPy::GetResource(const std::string &phase) {
  MS_LOG(DEBUG) << "Phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

FuncGraphPtr ExecutorPy::GetFuncGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->func_graph;
}

void ExecutorPy::SetJitPrimalFuncGraph(const FuncGraphPtr &primal_func_graph, const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal_func_graph->return_node())
      << "No executor info. found for phase: " << phase;
    return;
  }
  MS_EXCEPTION_IF_NULL(primal_func_graph);
  it->second->jit_primal_func_graph = primal_func_graph;
}

FuncGraphPtr ExecutorPy::GetJitPrimalFuncGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->jit_primal_func_graph;
}

FuncGraphPtr ExecutorPy::GetJitGradGraph(const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return it->second->jit_grad_graph;
}

void ExecutorPy::SetJitGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase) {
  const auto it = info_.find(phase);
  if (it == info_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, grad_graph->return_node()) << "No executor info. found for phase: " << phase;
    return;
  }
  if (it->second->jit_grad_graph != nullptr) {
    MS_LOG(DEBUG) << "The grad graph has existed, phase is: " << phase;
  }
  MS_EXCEPTION_IF_NULL(grad_graph);
  it->second->jit_grad_graph = grad_graph;
}

compile::VmEvalFuncPtr ExecutorPy::GetVmEvalFunc(const std::string &phase, const std::string &kind) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->HasResult(kind) && res->GetResult(kind).is<compile::VmEvalFuncPtr>()) {
    return res->GetResult(kind).cast<compile::VmEvalFuncPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kind:" << kind;
  return nullptr;
}

bool ExecutorPy::HasCompiled(const std::string &phase) const { return info_.count(phase) != 0; }

py::bytes ExecutorPy::GetFuncGraphProto(const std::string &phase, const std::string &ir_type, const bool &incremental) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  if (fg_ptr == nullptr) {
    for (const auto &item : info_) {
      MS_LOG(DEBUG) << "Phase key is: " << item.first;
    }
    MS_LOG(EXCEPTION) << "Can not find func graph " << phase;
  }

  if (ir_type == IR_TYPE_ANF) {
    std::string proto_str = GetFuncGraphProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ANF format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ONNX format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_MINDIR) {
    std::string proto_str = GetBinaryProtoString(fg_ptr, incremental);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export MINDIR format model failed.";
    }
    return proto_str;
  }

  MS_LOG(INTERNAL_EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::bytes GraphExecutorPy::GetOptimizeGraphProto(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "No phase in executor: " << phase;
  }
  FuncGraphPtr fg_ptr = info_[phase]->resource->optimize_graph();
  if (fg_ptr == nullptr) {
    MS_LOG(WARNING) << "Can not find optimize graph.";
    return "";
  }
  std::string proto_str = GetFuncGraphProtoString(fg_ptr);
  if (proto_str.empty()) {
    MS_LOG(EXCEPTION) << "Export optimize graph proto string failed.";
  }
  return proto_str;
}

void ExecutorPy::SetJitConfig(const py::dict &config) {
  auto jit_config = GenerateJitConfigMap(config);
  PhaseManager::GetInstance().set_jit_config(jit_config);
}

py::dict GraphExecutorPy::GetParallelGraphInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParallelGraphInfo!";
  std::string parallel_phase = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(parallel_phase);
  if (graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not access FuncGraph according to phase: " << parallel_phase;
  }

  return mindspore::parallel::GetParallelCNodeInfoFromGraph(graph);
}

py::dict GraphExecutorPy::GetParameterLayout(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  if (graph == nullptr) {
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParameterLayoutFromResource(resource);
  }
  return mindspore::parallel::GetParameterLayoutFromGraph(graph);
}

py::tuple GraphExecutorPy::FlopsCollection(const std::string &phase) {
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::FlopsCollection(graph);
}

py::dict GraphExecutorPy::GetCNodeStrategy(const std::string &phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  return stra_dict_[phase];
}

py::list GraphExecutorPy::GetParallelParameterNameList(const std::string &phase) {
  std::string param_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(param_graph);
  if (graph == nullptr) {
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParallelParameterNameListFromResource(resource);
  }
  return mindspore::parallel::GetParallelParameterNameListFromGraph(graph);
}

void GraphExecutorPy::SetCNodeStrategy(const std::string &name, const parallel::Strategies &strategy) {
  MS_LOG(DEBUG) << "SetCNodeStrategy!";
  stra_dict_[phase_][py::str(name)] = strategy;
}

size_t GraphExecutorPy::GetNumOpsInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetNumOpsInfo!";
  return phase_to_num_op_info_[phase];
}

void GraphExecutorPy::SetNumOpsInfo(size_t num_ops) {
  MS_LOG(DEBUG) << "SetNumOpsInfo!";
  phase_to_num_op_info_[phase_] = num_ops;
}

py::dict GraphExecutorPy::GetAllreduceFusion(const std::string &phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

// Not support multi thread, not support nested call too.
// Here using nested_called flg to avoid nested call.
void ExecutorPy::DelNetRes(const py::object &source, const py::set &id) {
  // no need to del net res by gc in independent dataset process which is a subprocess forked by main process
  if (process_id_ != GetCurrentPID()) {
    return;
  }
  ClearArgCache(source);
  // Del all graphs by different phase
  for (auto item : id) {
    DelOneNetRes(item);
  }
}

void GraphExecutorPy::DelOneNetRes(const py::handle &py_phase) {
  if (!pybind11::isinstance<py::str>(py_phase)) {
    MS_LOG(ERROR) << "Expect string phase, but got " << py::str(py_phase);
    return;
  }
  auto phase = pybind11::cast<std::string>(py_phase);
  MS_LOG(INFO) << "Delete one net resource start, phase: " << phase;
  auto iter = info_.find(phase);
  auto clear = false;
  if (iter != info_.end()) {
    clear = true;
    auto res = iter->second->resource;
    if (res->HasResult(kStepParallelGraph)) {
      std::string layout_graph = phase + kStepParallelGraph;
      (void)info_.erase(layout_graph);
    }
    (void)info_.erase(phase);
    MS_LOG(DEBUG) << "Delete phase: " << phase << ", info size: " << info_.size();
  }
  if (clear) {
    // Do clear here to avoid any pointer for resource.
    FuncGraphLoopBreaker::Inst().ClearCellGraphs(phase);
    FuncGraphLoopBreaker::Inst().CleanUnusedFuncGraphs(phase);
  }
  MS_LOG(INFO) << "Delete one net resource end. " << clear;
}

void GraphExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean Graph executor resource!";
  executor_ = nullptr;
}

void ExecutorPy::ClearInfo() {
  MS_LOG(INFO) << "Clean graph resource!";
  for (auto &item : info_) {
    if (item.second && item.second->resource) {
      item.second->resource->CleanBackend();
    }
  }
}

void ExecutorPy::set_process_id() { process_id_ = GetCurrentPID(); }

std::string ExecutorPy::get_queue_name(const std::string &dataset_phase) {
  return CompileCacheManager::GetCachedDataQueueName(dataset_phase);
}

GraphExecutorPy::~GraphExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

void GraphExecutorPy::SaveCompiledGraph(const std::string &phase) {
  // save the graph to GraphExecutorPy
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  info_[phase]->func_graph = func_graph;
  func_graph->set_attr("phase", MakeValue(GetPhasePrefix(phase)));

  if ((func_graph != nullptr) && parallel::IsAutoParallelCareGraph(func_graph)) {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph!";
    auto res = info_[phase]->resource;
    // When using frontend compile cache, model parallel parameter layout graph is not saved.
    if (res->HasResult(kStepParallelGraph)) {
      func_graph = res->GetResult(kStepParallelGraph).cast<FuncGraphPtr>();
      ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
      std::string layout_graph = phase + kStepParallelGraph;
      executor_info->func_graph = func_graph;
      info_[layout_graph] = executor_info;
    }
  } else {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "End save compiled func graph!";
}

bool IsPhaseExportAir(const std::string &phase) {
  auto phase_to_export = "export.air";
  return phase.rfind(phase_to_export) != std::string::npos;
}

bool IsPhaseExport(const std::string &phase) {
  constexpr auto export_str = "export";
  return phase.compare(0, strlen(export_str), export_str) == 0;
}

bool IsPhaseTrain(const std::string &phase) {
  const std::string phase_to_train = "train";
  return phase.rfind(phase_to_train) != std::string::npos;
}

bool IsPhaseLoadFromMindIR(const std::string &phase) {
  const std::string mindir_graph = "graph_load_from_mindir";
  return phase.rfind(mindir_graph) != std::string::npos;
}

std::vector<ActionItem> GetActions(const ResourcePtr &resource, const std::string &phase, bool trace_flag = false,
                                   bool erase_parse = false) {
  MS_EXCEPTION_IF_NULL(resource);
  compile::SetMindRTEnable();
  return VmPipeline(resource, trace_flag, erase_parse);
}

void ExecutorPy::InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase) {
  // The compilation cache only support for training cell or functions decorated with 'jit' currently.
  // If enable compilation cache, it will get a non-empty dependent files list from python.
  if (!CompileCacheEnable()) {
    return;
  }
  bool has_python_script = true;
  if (compile_cache_dep_files_.empty()) {
    has_python_script = false;
  }

  {
    MsProfileStatGuard stat_guard("LoadCachedFuncGraph");
    static size_t idx = 0;
    MS_EXCEPTION_IF_NULL(resource);
    resource->GetCompileCacheResource(compile_cache_dep_files_, weights_, queue_name_, idx++,
                                      &compile_cache_consistent_, has_python_script);
    if (resource->func_graph() != nullptr) {
      ResetId(resource);
    }
  }
}

void ExecutorPy::InitCompileCacheResource(const ResourcePtr &resource, const std::string &phase) {
  InitCompileCacheInfo(resource, phase);
  bool enable_compile_cache = resource->EnableCompileCache();
  bool use_compile_cache = enable_compile_cache && resource->func_graph();
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  compile_cache_context.SetUseCompileCache(use_compile_cache);
  ConfigManager::GetInstance().ResetQueue(queue_name_);
}

void GraphExecutorPy::ParallelPostProcess(const std::string &phase, bool use_compile_cache) {
  // Slice Python parameter obj
  auto layout_graph = phase + kStepParallelGraph;
  // only Parallel graph has tensor_layout
  auto root = GetFuncGraph(layout_graph);
  bool after_shard = false;
  if (phase.find("after_shard") != std::string::npos) {
    after_shard = true;
  }
  // Use compile cache
  if (use_compile_cache) {
    parallel::InitCompileCacheParams(info_[phase]->resource);
    return;
  }
  // Initialize parameters for graph which auto-parallel not care.
  if (root == nullptr && !after_shard) {
    auto graph = info_[phase]->resource->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    parallel::InitPynativeNoShardParams(graph);
    return;
  }
  MS_EXCEPTION_IF_NULL(root);
  parallel::AutoParallelPostProcess(root);
}

// Clean all resource not used in the future and cache generated during compiling.
void GraphExecutorPy::CleanCompileRes(const ResourcePtr &resource) {
  MS_LOG(INFO) << "Clean compile resource start";
  parallel::ParallelContext::GetInstance()->set_dynamic_shape_parallel_flag(false);
  ProcessStatus::GetInstance().RecordStart(kPipelineClean);
  uint64_t start_time = profiler::GetClockSyscnt();
  abstract::AnalysisContext::ClearContext();
  ClearCompileArgumentsResource();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  ad::g_k_prims.clear();
  ad::DFunctor::Clear();
  ReclaimOptimizer();
  resource->Clean();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  if (parallel_context->hccl_test_available()) {
    parallel::g_device_manager = nullptr;
  }
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  (void)profiler::CollectHostInfo(kCompiler, kPipelineClean, kPipelineClean, start_time, profiler::GetClockSyscnt(), 0);
  ProcessStatus::GetInstance().RecordEnd();
  CompileCacheContext::GetInstance().Clear();
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "Clean compile resource end";
}

bool GraphExecutorPy::CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs,
                                   const std::string &phase, bool trace_flag) {
  JitCompilingScope jit_compiling_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  PhaseManager::GetInstance().set_phase(phase);
  phase_ = phase;

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>();
  resource->set_func_graph(graph);
  InitCompileCacheResource(resource, phase_);

  bool erase_parse = true;
  auto actions = GetActions(resource, phase, trace_flag, erase_parse);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, actions);

  if (pip->NeedCreateBackend()) {
    // Create backend asynchronously.
    resource->SetBackendAsync([]() {
      auto backend = compile::CreateBackend();
#ifdef ENABLE_DEBUGGER
      // Connect session to debugger.
      backend->SetDebugger();
#endif
      return backend;
    });
  }

  // Get the parameters items and add the value to args_abs.
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  bool is_auto_parallel = (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                           parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel);
  ConvertArgs(args, kwargs, is_auto_parallel, &args_abs, &arguments);
  ConvertSymbolicShape(args, &args_abs);
  AddManagerForFuncGraphArgs(resource, arguments);
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  executor_info->arg_list_size = args.size() + kwargs.size();
  executor_info->resource = resource;
  info_[phase] = executor_info;
  pip->Run();

  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase);
  if (is_auto_parallel) {
    bool use_compile_cache = resource->EnableCompileCache() && resource->func_graph();
    ParallelPostProcess(phase, use_compile_cache);
  }
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
#endif
  CleanCompileRes(resource);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

bool GraphExecutorPy::CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                                   const py::object &phase) {
  JitCompilingScope jit_compiling_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  // Check if the phase is valid.
  if ((!py::isinstance<py::str>(phase))) {
    MS_LOG(ERROR) << "The `phase` must be string.";
    return false;
  }
  // Check if the function or net is valid.
  if (py::isinstance<py::none>(source)) {
    MS_LOG(ERROR) << "The source object to compile should not be None.";
    return false;
  }
  // Check if the args of function or net is valid.
  CheckArgsValid(source, args);

  source_ = py::cast<std::string>(py::str(source));
  phase_ = py::cast<std::string>(phase);
  PhaseManager::GetInstance().set_phase(phase_);
  obj_desc_ = GetObjDesc(source);
  MS_LOG(INFO) << "Start compiling, phase: " << phase_;

  auto root_func_name = obj_desc_;
  std::replace(root_func_name.begin(), root_func_name.end(), '.', '_');
  std::replace(root_func_name.begin(), root_func_name.end(), '\'', '_');
  opt::LoadPassesConfig(root_func_name);

  PROF_START(compile_graph);
  MS_LOG(DEBUG) << "source: {" << source_ << "}\nargs: " << py::str(const_cast<py::tuple &>(args))
                << "\nkwargs: " << py::str(const_cast<py::dict &>(kwargs));
  EventMessage::PrintCompileStartMsg(phase_, obj_desc_);

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>(source);
  InitCompileCacheResource(resource, phase_);

  auto actions = GetActions(resource, phase_, false, false);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, actions);

  uint64_t start_time = profiler::GetClockSyscnt();
  if (pip->NeedCreateBackend()) {
    // Create backend asynchronously.
    resource->SetBackendAsync([]() {
      auto backend = compile::CreateBackend();
#ifdef ENABLE_DEBUGGER
      // Connect session to debugger.
      backend->SetDebugger();
#endif
      return backend;
    });
  }
  (void)profiler::CollectHostInfo(kCompiler, kCreateBackend, kCreateBackend, start_time, profiler::GetClockSyscnt(), 0);

  // Get the parameters items and add the value to args_abs.
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  bool is_parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                          parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel;
  bool is_auto_parallel = is_parallel_mode && !py::hasattr(source, parallel::kSkipAutoParallelCompile) &&
                          !py::hasattr(source, parallel::kKeepInputUnchanged);
  ConvertArgs(args, kwargs, is_auto_parallel, &args_abs, &arguments);
  ConvertSymbolicShape(args, &args_abs);
  AddManagerForFuncGraphArgs(resource, arguments);
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  executor_info->arg_list_size = args.size() + kwargs.size();
  executor_info->resource = resource;
  info_[phase_] = executor_info;
  pip->Run();

  opt::SavePassesConfig(root_func_name);
  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase_);
  PROF_START(ParallelPostProcess);
  if (is_parallel_mode) {
    bool use_compile_cache = resource->EnableCompileCache() && resource->func_graph();
    ParallelPostProcess(phase_, use_compile_cache);
  }
  PROF_END(ParallelPostProcess);
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
#endif
  PROF_START(CleanCompileRes);
  CleanCompileRes(resource);
  EventMessage::PrintCompileEndMsg(phase_, obj_desc_);
  PhaseManager::GetInstance().ClearPhase();
  PROF_END(CleanCompileRes);
  MS_LOG(INFO) << "Finish compiling.";
  PROF_END(compile_graph);
  return true;
}

void SetHookForArgAbstract(const py::object &arg, abstract::AbstractBasePtr abs) {
  if (py::isinstance<Tensor>(arg)) {
    auto tensor = arg.cast<std::shared_ptr<Tensor>>();
    if (tensor->has_user_data("backward_hook") && !abs->has_user_data("backward_hook")) {
      MS_LOG(DEBUG) << "set hooks for arg: " << py::str(arg) << ", abs(" << abs.get() << "): " << abs << ".";
      auto hook_map = tensor->user_data<std::map<uint64_t, py::function>>("backward_hook");
      auto hook_fns = std::make_shared<std::vector<py::function>>();
      for (auto iter = hook_map->begin(); iter != hook_map->end(); iter++) {
        hook_fns->push_back(iter->second);
      }
      abs->set_user_data("backward_hook", hook_fns);
    }
  } else {
    MS_LOG(DEBUG) << "arg: " << py::str(arg) << " is not a Tensor, we only support arg of type Tensor now.";
  }
}

void GraphExecutorPy::ConvertArgs(const py::tuple &args, const py::dict &kwargs, bool is_auto_parallel,
                                  abstract::AbstractBasePtrList *args_abs, std::vector<ValuePtr> *arguments) {
  MS_EXCEPTION_IF_NULL(args_abs);
  MS_EXCEPTION_IF_NULL(arguments);
  for (std::size_t i = 0; i < args.size(); i++) {
    // In some parallel mode need full_tensor which cause the args of GenerateArgumentsKey not same to compile,
    // So can't use cur_convert_input_ directly.
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments->emplace_back(iter->second.first);
      if (is_auto_parallel) {
        auto abs_item = iter->second.second->Clone();
        (void)parallel::ExtendInputArgsAbstractShape(abs_item, i);
        (void)args_abs->emplace_back(abs_item);
        continue;
      }
      (void)args_abs->emplace_back(iter->second.second);
      SetHookForArgAbstract(args[i], iter->second.second);
      continue;
    }
    ValuePtr converted = nullptr;
    bool success = parse::ConvertData(args[i], &converted);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i
                                 << "]: " << py::str(args[i]);
    }
    (void)arguments->emplace_back(converted);
    auto args_abstract_item = ArgsToAbstract(args[i], converted, enable_tuple_broaden_);
    if (is_auto_parallel) {
      (void)parallel::ExtendInputArgsAbstractShape(args_abstract_item, i);
    }
    (void)args_abs->emplace_back(args_abstract_item);
    SetHookForArgAbstract(args[i], args_abstract_item);
  }
  for (const auto &item : kwargs) {
    auto iter = cur_convert_input_.find(item.first.ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments->emplace_back(iter->second.first);
      (void)args_abs->emplace_back(iter->second.second);
      auto keyword_arg_abs = iter->second.second->cast<abstract::AbstractKeywordArgPtr>();
      SetHookForArgAbstract(py::cast<py::object>(item.second), keyword_arg_abs->get_arg());
      continue;
    }
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.first), &key) &&
                   parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the argument (" << py::str(item.first) << ": "
                                 << py::str(item.second) << ").";
    }
    AbstractBasePtr value_abs = ArgsToAbstract(py::cast<py::object>(item.second), value, enable_tuple_broaden_);
    auto keyword_arg_abs = std::make_shared<abstract::AbstractKeywordArg>(GetValue<std::string>(key), value_abs);
    (void)arguments->emplace_back(value);
    (void)args_abs->emplace_back(keyword_arg_abs);
    SetHookForArgAbstract(py::cast<py::object>(item.second), value_abs);
  }
}

void GraphExecutorPy::ConvertSymbolicShape(const py::tuple &args, AbstractBasePtrList *args_abs) {
  std::vector<symshape::SymbolInfoList> symbol_infos;
  symbol_infos.reserve(args_abs->size());
  bool has_dyn_shape = false;
  bool is_parallel = parallel::IsSemiOrAutoParallelMode();

  for (size_t i = 0; i < args.size(); i++) {
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter == cur_convert_input_.end()) {
      continue;
    }
    auto &info_list = symbol_infos.emplace_back(symshape::SymbolInfoList{});
    if (!iter->second.first->isa<MetaTensor>()) {
      continue;
    }
    auto digital_shape = iter->second.second->GetShape();
    MS_EXCEPTION_IF_NULL(digital_shape);
    if (digital_shape->IsDynamic()) {
      has_dyn_shape = true;
    }
    constexpr char symbolic_shape_attr[] = "symbolic_shape";
    if (!py::hasattr(args[i], symbolic_shape_attr)) {
      if (is_parallel && digital_shape->isa<abstract::TensorShape>()) {
        info_list.resize(digital_shape->GetShapeVector().size());
      }
      continue;
    }
    auto symbolic_shape_obj = py::getattr(args[i], symbolic_shape_attr);
    MS_EXCEPTION_IF_CHECK_FAIL(py::isinstance<py::list>(symbolic_shape_obj), "tensor.symbolic_shape should be a list");
    auto obj_list = py::cast<py::list>(symbolic_shape_obj);
    info_list.resize(obj_list.size());
    for (size_t j = 0; j < obj_list.size(); j++) {
      if (!py::isinstance<py::dict>(obj_list[j])) {
        continue;
      }
      auto dict_obj = py::cast<py::dict>(obj_list[j]);
      for (auto cfg_iter = dict_obj.begin(); cfg_iter != dict_obj.end(); ++cfg_iter) {
        auto cfg_key = py::cast<std::string>(cfg_iter->first);
        if (cfg_key == "max") {
          info_list[j].max = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "min") {
          info_list[j].min = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "divisor") {
          info_list[j].divisor = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "remainder") {
          info_list[j].remainder = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "id") {
          info_list[j].id = py::cast<int64_t>(cfg_iter->second);
        } else if (cfg_key == "name") {
          info_list[j].name = py::cast<std::string>(cfg_iter->second);
        }
      }
    }
  }

  MS_LOG(DEBUG) << "before parallel symbol";
  parallel::PrintSymbolInfo(symbol_infos);
  symbol_infos = parallel::ParallelSymbolInfo(symbol_infos, has_dyn_shape);
  MS_LOG(DEBUG) << "after parallel symbol";
  parallel::PrintSymbolInfo(symbol_infos);

  auto symbolic_shape_list = symshape::BuildSymbolicShapeBySymbolInfo(*args_abs, symbol_infos);
  for (size_t i = 0; i < symbolic_shape_list.size(); i++) {
    // when the same tensor object is used in set_inputs interface, the inputs may shared a same Abstract object.
    // but for dynamic shape, the same "-1" in abstract can be different symbolic shape.
    auto abs = symshape::CloneAbstractIfSymbolExists((*args_abs)[i]);
    MS_EXCEPTION_IF_NULL(abs);
    abs->SetSymbolicShape(symbolic_shape_list[i]);
    (*args_abs)[i] = abs;
  }
}

void ExecutorPy::ReleaseResourceOnException(const py::object &phase) {
  bool clear = false;
  // Be sure the pointer res destroyed before do DelOneNetRes.
  {
    ResourcePtr res = GetResource(py::cast<std::string>(phase));
    if (res != nullptr) {
      clear = true;
      CleanCompileRes(res);
    }
  }
  ProcessStatus::GetInstance().Clear();
  if (clear) {
    DelOneNetRes(phase);
  }
}

bool ExecutorPy::Compile(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                         const py::object &phase) {
  bool res = false;
  HandleExceptionRethrow(
    [this, &res, &source, &args, &kwargs, &phase]() {
      bool executor_running = false;
      std::string running_obj_desc;
      if (GraphExecutorPy::GetInstance()->executor_running()) {
        executor_running = true;
        running_obj_desc = GraphExecutorPy::GetInstance()->obj_desc();
      } else if (JitExecutorPy::GetInstance()->executor_running()) {
        executor_running = true;
        running_obj_desc = JitExecutorPy::GetInstance()->obj_desc();
      }
      if (executor_running) {
        MS_LOG(EXCEPTION) << "Nested execution during JIT execution for " << GetObjDesc(source) << " is not supported "
                          << "when " << running_obj_desc << " compile and execute. For more details, please refer to "
                          << "https://www.mindspore.cn/search?inputValue=Nested%20execution";
      }
      ProcessStatus::GetInstance().RecordStart(kCompiler);
      std::map<std::string, std::string> custom_info;
      custom_info["phase"] = py::cast<std::string>(phase);
      uint64_t start_time = profiler::GetClockSyscnt();
      res = CompileInner(source, args, kwargs, phase);
      (void)profiler::CollectHostInfo(kCompiler, kCompiler, kCompiler, start_time, profiler::GetClockSyscnt(), 1,
                                      custom_info);
      ProcessStatus::GetInstance().RecordEnd();
      ProcessStatus::GetInstance().Print();
    },
    [this, &phase]() {
      if (!StaticAnalysisException::Instance().HasException()) {
        // print function call stack info before release
        std::string compile_exception_info = GetCompileExceptionInfo();
        if (!compile_exception_info.empty()) {
          MS_LOG(ERROR) << compile_exception_info;
        }
      }
      ReleaseResourceOnException(phase);
    },
    [this, &phase]() { ReleaseResourceOnException(phase); }, [this, &phase]() { ReleaseResourceOnException(phase); });

  // Set need recompile to false after compile finished.
  return res;
}

void CacheFuncGraph(const ResourcePtr &resource) {
  if (!resource->EnableCompileCache()) {
    return;
  }
  {
    MsProfileStatGuard stat_guard("SaveCacheFuncGraph", "compile_cache", true);
    resource->CacheFuncGraph();
  }
}

void CheckInterpretNodeLineInfos() {
  auto &py_interpret_nodes = InterpretNodeRecorder::GetInstance().PyInterpretNodes();
  auto &py_execute_nodes = InterpretNodeRecorder::GetInstance().PyExecuteNodes();
  if (py_interpret_nodes.empty() && py_execute_nodes.empty()) {
    return;
  }

  std::stringstream ss;
  ss << "Found unsupported syntax in graph mode, those codes would be fallen back to Python interpreter:\n";
  // Dump for PyInterpret.
  ss << "----------------------------------------\n";
  ss << " After Parser Phase (total: " << py_interpret_nodes.size() << ")\n";
  ss << "----------------------------------------\n";
  size_t num = 1;
  for (const auto &node : py_interpret_nodes) {
    const auto line_info = trace::GetDebugInfoStr(node->debug_info());
    ss << "# No. " << num << ":\n" << line_info << "\n";
    ++num;
  }
  ss << "\n";
  // Dump for PyExecute.
  ss << "----------------------------------------\n";
  ss << " After Optimizer Phase (total: " << py_execute_nodes.size() << ")\n";
  ss << "----------------------------------------\n";
  num = 1;
  for (const auto &node : py_execute_nodes) {
    ss << "# No. " << num << ":\n";
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &weak_script_node = cnode->weak_input(1);
    const auto &script_node = weak_script_node.lock();
    MS_EXCEPTION_IF_NULL(script_node);
    const auto &script = GetValueNode<StringImmPtr>(script_node);
    // Usually the script is a value node.
    std::string script_str;
    if (script != nullptr) {
      script_str = script->value();
    } else {
      const auto &script_abs = script_node->abstract();
      if (script_abs != nullptr) {
        const auto script_abs_scalar = script_abs->cast<abstract::AbstractScalarPtr>();
        auto script_value = script_abs_scalar->BuildValue();
        MS_EXCEPTION_IF_NULL(script_value);
        auto script_value_str = script_value->cast<StringImmPtr>();
        MS_EXCEPTION_IF_NULL(script_value_str);
        script_str = script_value_str->value();
      }
    }
    if (!script_str.empty()) {
      ss << "Script: " << script_str << "\n\n";
    } else {
      ss << "Node: " << node->DebugString() << "\n\n";
    }
    const auto line_info = trace::GetDebugInfoStr(node->debug_info());
    ss << line_info << "\n";
    ++num;
  }
  ss << "\n";
  ss << "----------------------------------------\n";

  // Print the codes run in JIT Fallback.
  if (common::GetEnv("MS_DEV_FALLBACK_DUMP_NODE") == "1") {
    MS_LOG(ERROR) << ss.str();
  } else {
    MS_LOG(INFO) << ss.str();
  }
  InterpretNodeRecorder::GetInstance().Clear();
}

#ifdef ENABLE_DUMP_IR
void RDRRecordGraph(const size_t action_index, const size_t action_size, const std::string &filename,
                    const FuncGraphPtr &graph) {
  if (mindspore::RecorderManager::Instance().RdrEnable()) {
    MS_LOG(INFO) << "Recording FuncGraph in pipeline using RDR.";
    if (graph != nullptr) {
      auto graph_clone = BasicClone(graph);
      if (graph_clone != nullptr) {
        DumpGraphParams dump_params = {false, static_cast<int>(kTopStack)};
        if (action_index == action_size) {
          dump_params.dump_mode = static_cast<int>(kWholeStack);
        }
        (void)mindspore::RDR::RecordAnfGraph(SUBMODULE_ID, filename, graph_clone, dump_params, ".ir");
      } else {
        MS_LOG(WARNING) << "Clone FuncGraph failed in pipeline, no FuncGraph recording in RDR.";
      }
    } else {
      MS_LOG(WARNING) << "Pipeline Resource has no FuncGraph, no FuncGraph recording in RDR";
    }
    MS_LOG(INFO) << "Recording FuncGraph in pipeline end.";
  }
}
#endif

#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name) {
  std::ostringstream oss;
  int spaces = 2;
  oss << std::setfill('0') << std::setw(spaces) << stage_idx << "_" << action_name;
  return oss.str();
}

void RecordIR(const size_t action_index, const size_t action_size, const std::string &action_name,
              const FuncGraphPtr &graph, FuncGraphPtr *user_graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory) && graph != nullptr) {
    *user_graph = graph;
    std::string base_name = GetBaseNameForIR(SizeToLong(action_index), action_name);

    // Generate IR file in human-readable format
    static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
    if (switch_order) {
      ExportIR(base_name + ".ir", graph);
    } else {
      DumpIR(base_name + ".ir", graph, true, kWholeStack);
    }
    if (context->CanDump(kFully)) {
      draw::Draw(base_name + ".dot", graph);
    }
  }
}
#endif

void SaveGraphForReadability(const std::string &action_name, const FuncGraphPtr &graph, const ResourcePtr &resource) {
  if (graph != nullptr && action_name.find("optimize") != string::npos) {
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIRProto(graph, action_name);
    }
#endif
    resource->set_optimize_graph(graph);
  }
}

void Pipeline::Run() {
  MS_LOG(INFO) << "Pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;
  const std::string last_compile_action = kValidate;
  bool already_print_profile = false;
  static const auto compile_profile_finish_action = common::GetCompileConfig("COMPILE_PROFILE_FINISH_ACTION");
  ProfileExecute(MsProfile::GetProfile(), [this, &user_graph, &last_compile_action, &already_print_profile]() {
    size_t i = 0;
    for (auto &action : actions_) {
      std::string action_name = action.first;
      MsProfileStatGuard stat_guard(std::move(action_name), "compile_action", true);
#ifdef ENABLE_TIMELINE
      DumpTime &dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      ProcessStatus::GetInstance().RecordStart(action.first);
      uint64_t start_time = profiler::GetClockSyscnt();
      bool result = true;
      ProfileExecute(MsProfile::GetProfile()->Step(action.first), [&result, &action, this]() {
        MS_LOG(INFO) << "Status record: start " << action.first << " action.";
        result = action.second(resource_);
        MS_LOG(INFO) << "Status record: end " << action.first << " action.";
        if (IS_OUTPUT_ON(mindspore::kInfo)) {
          auto func_graph = resource_->func_graph();
          MS_EXCEPTION_IF_NULL(func_graph);
          auto manager = func_graph->manager();
          MS_EXCEPTION_IF_NULL(manager);
          MS_LOG(INFO) << "Extra status record: total func graphs: " << manager->func_graphs().size()
                       << ", total nodes: " << manager->all_nodes().size();
        }
      });
      (void)profiler::CollectHostInfo(kCompiler, action.first, action.first, start_time, profiler::GetClockSyscnt(), 0);
      ProcessStatus::GetInstance().RecordEnd();
      if (!result) {
        MS_LOG(INTERNAL_EXCEPTION) << "Pipeline running to end, failed in step:" << action.first;
      }

      if (EnabledProfile() && compile_profile_finish_action == action.first) {
        ProfileExecuteBreak(MsProfile::GetProfile());
        MsProfile::Print();
        already_print_profile = true;
      }

      if (action.first == kTaskEmit) {
        SetLoopCount(resource_);
      } else if (action.first == last_compile_action) {
        CheckInterpretNodeLineInfos();
        CacheFuncGraph(resource_);
#ifdef WITH_BACKEND
        MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
        if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
          const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
            {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
          MS_EXCEPTION_IF_NULL(device_context);
          MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
          device_context->GetDeprecatedInterface()->DumpProfileParallelStrategy(resource_->func_graph());
        }
#endif
        ResetId(resource_);
      }
      FuncGraphPtr graph = resource_->func_graph();
#ifdef ENABLE_DUMP_IR
      std::string filename = GetBaseNameForIR(SizeToLong(i), action.first);
      RDRRecordGraph(i, actions_.size(), filename, graph);
      RecordIR(i, actions_.size(), action.first, graph, &user_graph);
#endif
      SaveGraphForReadability(action.first, graph, resource_);
      i++;
#ifdef ENABLE_TIMELINE
      dump_time.Record(action.first, GetTime(), false);
#endif
    }
  });

  if (EnabledProfile()) {
    if (!already_print_profile) {
      MsProfile::Print();
    }
    MsProfile::Reset();
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory) && (user_graph != nullptr)) {
    if (context->CanDump(kFully)) {
      draw::DrawUserFuncGraph("ModelDigraph.dot", user_graph);
    }
  }
  if (common::GetEnv("DUMP_PARALLEL_INFO") == "1") {
    std::unordered_map<std::string, std::vector<uint32_t>> group_map;
    if (distributed::collective::CollectiveManager::instance()->initialized()) {
      group_map = distributed::collective::CollectiveManager::instance()->get_group_map();
    }
    if (parallel::g_device_manager == nullptr) {
      MS_LOG(WARNING) << "parallel::g_device_manager is not initialized. Skip dump parallel info.";
    } else {
      auto global_rank_id = parallel::g_device_manager->global_rank();
      DumpParallelJson("dump_parallel_info_" + std::to_string(global_rank_id) + ".json", resource_->func_graph(),
                       global_rank_id, group_map);
    }
  }
#endif
  MS_LOG(INFO) << "End";
}

bool Pipeline::NeedCreateBackend() {
  return std::any_of(actions_.begin(), actions_.end(),
                     [](const ActionItem &action) { return action.first == kTaskEmit || action.first == kExecute; });
}

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  bool arg_list_inited = !arg_list->empty();
  for (std::size_t i = 0; i < args.size(); i++) {
    py::object arg = args[i];
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(arg, &converted);
    if (!succ) {
      MS_LOG(INTERNAL_EXCEPTION) << "The " << i << "th arg convert failed.";
    }
    if (!arg_list_inited) {
      arg_list->push_back(converted);
      continue;
    }
    if (i >= arg_list->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "i:" << i << " output of range:" << arg_list->size();
    }
    (*arg_list)[i] = converted;
  }

  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  const std::vector<AnfNodePtr> &graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // Maybe some default parameter
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG_WITH_NODE(EXCEPTION, graph_params[i]) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<Tensor>()) {
        MS_LOG_WITH_NODE(EXCEPTION, graph_params[i])
          << "Parameter[" << param_ptr->ToString() << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

void ExecutorPy::ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kGraphExecutorPy, runtime::ProfilerEvent::kInputProcess,
                                     phase);
  ProcessVmArgInner(args, GetResource(phase), arg_list);
}

#ifdef ENABLE_DEBUGGER
void GraphExecutorPy::TerminateDebugger() {
  if (Common::GetDebugTerminate()) {
    MS_LOG(INFO) << "Terminate debugger and clear resources!";
    ClearResAtexit();
    exit(static_cast<int>(!Common::GetDebugExitSuccess()));
  }
}
#endif

py::object ExecutorPy::Run(const py::tuple &args, const py::object &phase) {
  py::object res;
  HandleExceptionRethrow(
    [this, &res, &args, &phase]() {
      executor_running_ = true;

      uint64_t start_time = 0;
      PROFILER_START(start_time);
      res = RunInner(args, phase);
      PROFILER_STAGE_END(start_time, runtime::ProfilerStage::kRunGraph);

      executor_running_ = false;
    },
    [this]() { executor_running_ = false; }, [this]() { executor_running_ = false; },
    [this]() { executor_running_ = false; }, nullptr, true);
  return res;
}

void ExecutorPy::ClearRunArgumentsResource(size_t input_arg_size, VectorRef *arg_list) {
  for (std::size_t i = 0; i < input_arg_size; ++i) {
    (*arg_list)[i] = nullptr;
  }
}

py::object GraphExecutorPy::RunInner(const py::tuple &args, const py::object &phase_obj) {
  JitRunningScope jit_running_scope;
  if (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileGraph) {
    py::int_ ret = 0;
    return ret;
  }
  // Mindspore debugger notify main thread to exit after one step, and will not run next step
#ifdef ENABLE_DEBUGGER
  TerminateDebugger();
#endif
  if (!py::isinstance<py::str>(phase_obj)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase = py::cast<std::string>(phase_obj);
  auto phase_prefix = GetPhasePrefix(phase);
  PhaseManager::GetInstance().set_phase(phase_prefix);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    PhaseManager::GetInstance().set_phase(phase);
  }
#ifdef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    if (phase_prefix == "save") {
      phase.erase(0, kPhaseSavePrefixLen);
      compile::VmEvalFuncPtr run = GetVmEvalFunc("train." + phase, kCkptOutput);
      if (run == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "Can't find run graph func for " << phase;
      }

      VectorRef ckpt_args;
      (void)(*run)(ckpt_args);
      ConfigManager::GetInstance().ResetConfig();
      return py::none();
    }
  }
#endif
  auto ret_val = std::make_shared<py::object>();
  if (info_.count(phase) != 0 && info_[phase]->func_graph != nullptr) {
    if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
      return *ret_val;
    }
  }
#ifndef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    // Virtual output constructed for test cases.
    if (!args.empty()) {
      return args[0];
    }
    return args;
  }
#endif
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  auto &execute_info = iter->second;
  MS_EXCEPTION_IF_NULL(execute_info);
  if (args.size() > execute_info->arg_list_size) {
    MS_LOG(WARNING) << "The args size: " << args.size() << ", full_arg_size: " << execute_info->arg_list_size;
  }
  ProcessVmArg(args, phase, &execute_info->arg_list);
  // Start to run phase.
  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase);
  if (run == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can't find run graph func for " << phase;
  }

  MS_LOG(DEBUG) << "Eval run " << ms_context->backend_policy();
  const auto &output = execute_info->func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  const auto &output_abs = output->abstract();
  MS_EXCEPTION_IF_NULL(output_abs);
  BaseRef value = (*run)(execute_info->arg_list);
  bool need_recovery = distributed::recovery::RecoveryContext::GetInstance()->enable_recovery() &&
                       distributed::recovery::RecoveryContext::GetInstance()->need_reset();
  if (need_recovery) {
    // In recovery scenario, the output value could be empty, do not transform return data.
    return py::none();
  }
  py::object res = BaseRefToPyDataWithUserData(value, output_abs);
  ClearRunArgumentsResource(args.size(), &execute_info->arg_list);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(DEBUG) << "Run end";
  return res;
}

FuncGraphPtr GraphExecutorPy::BuildGraph(const py::dict &init_params, const std::string &phase) const {
  MS_LOG(INFO) << "Start build df graph, phase = " << phase;
  if (info_.count(phase) == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "No phase in executor: " << GetPhasePrefix(phase);
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target != kAscendDevice) {
    MS_LOG(INFO) << "Only Support ascend.";
    return nullptr;
  }

  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Phase " << phase << " must compile.";
  }
  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
  MS_EXCEPTION_IF_NULL(mindrt_backend);
  std::map<std::string, std::shared_ptr<Tensor>> init_tensors{};
  ConvertObjectToTensors(mindrt_backend, init_params, &init_tensors, info_.at(phase)->func_graph);
  return mindrt_backend->BuildDFGraph(info_.at(phase)->func_graph, init_tensors);
}

void GraphExecutorPy::ConvertObjectToTensors(const std::shared_ptr<compile::MindRTBackend> &backend,
                                             const py::dict &dict,
                                             std::map<std::string, std::shared_ptr<Tensor>> *const tensors,
                                             const FuncGraphPtr &anf_graph) const {
  const auto &infer_need_update_parameter_names = backend->GetInferParameterNames();
  bool infer = false;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool enable_ge = context_ptr->backend_policy() == "ge";
  bool is_train = false;
  if (anf_graph->has_attr("phase")) {
    std::string phase = anf_graph->get_attr("phase")->ToString();
    is_train = phase == "train";
  }
  if (enable_ge && !is_train) {
    infer = true;
  }
  for (auto item : dict) {
    if ((!py::isinstance<py::str>(item.first))) {
      MS_LOG(WARNING) << "Type of key of py_dict is not string, ignore it.";
      continue;
    }
    std::shared_ptr<Tensor> tensor;
    std::string name = py::cast<std::string>(item.first);

    if (infer && infer_need_update_parameter_names.find(name) == infer_need_update_parameter_names.end() &&
        !IsEnableRefMode()) {
      continue;
    }
    if (py::isinstance<py::float_>(item.second.attr("data"))) {
      // convert float to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeFloat32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<py::int_>(item.second.attr("data"))) {
      // convert int64_t to tensor with shape([1])
      tensor = std::make_shared<Tensor>(kNumberTypeInt32, std::vector<int64_t>({1}));
      *(static_cast<float *>(tensor->data_c())) = py::cast<float>(item.second.attr("data"));
    } else if (py::isinstance<Tensor>(item.second.attr("data"))) {
      // cast tensor
      tensor = py::cast<std::shared_ptr<Tensor>>(item.second.attr("data"));
    } else if (IsStubTensor(item.second.attr("data"))) {
      // cast stub_tensor
      tensor = ConvertStubTensor(item.second.attr("data"));
    }

    if (tensor == nullptr) {
      MS_LOG(EXCEPTION) << "Get default value for " << name << " failed";
    }
    (void)tensors->emplace(name, tensor);
  }
}

void GraphExecutorPy::UpdataParamNodeDefaultInput(
  const std::string &phase, const std::unordered_map<std::string, tensor::TensorPtr> &params_value) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "UpdataParamNodeDefaultInput for func graph(" << func_graph->ToString() << ") phase(" << phase
                << ")!";
  auto &params = func_graph->parameters();
  for (const auto &param : params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_cast = param->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(param_cast);
    auto iter = params_value.find(param_cast->name());
    if (iter != params_value.end()) {
      param_cast->set_default_param(iter->second);
    }
  }
}

py::dict ExecutorPy::GetParams(const std::string &phase) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  py::dict parameter_dict;
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  for (auto &param : graph_params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_ptr = std::static_pointer_cast<Parameter>(param);
    std::string name = param_ptr->name();
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_ptr->default_param());
    if (tensor != nullptr) {
      parameter_dict[py::str(name)] = *tensor;
    }
  }
  return parameter_dict;
}

py::bytes GraphExecutorPy::GetRandomStatus(const std::string &phase) const {
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(ERROR) << "Phase " << phase << " must compile.";
    return "";
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  MS_EXCEPTION_IF_NULL(iter->second->resource);
  auto &resource = iter->second->resource;
  auto backend = resource->GetBackend();
  const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
  MS_EXCEPTION_IF_NULL(mindrt_backend);
  auto actor_info = resource->GetResult(kActorInfo).cast<compile::ActorInfo>();
  auto random_status = mindrt_backend->GetRandomStatus(actor_info);
  return py::bytes(random_status.c_str(), random_status.size());
}

void GraphExecutorPy::PyExePath(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  auto py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void GraphExecutorPy::KernelBuildServerDir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  auto kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}
void GraphExecutorPy::SetOptimizeConfig(const py::list &optimize_cfg) {
  opt::PassConfigure::Instance().SetOptimizeConfig(optimize_cfg);
}
std::string GraphExecutorPy::GetOptimizeConfig() { return opt::PassConfigure::Instance().GetOptimizeConfig(); }
void GraphExecutorPy::SetConfigPasses(const py::list &passes) {
  opt::PassConfigure::Instance().SetConfigPasses(passes);
}
py::list GraphExecutorPy::GetRunningPasses() { return opt::PassConfigure::Instance().GetRunningPasses(); }

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &, bool need_run) {
  if (UseSimulationApi()) {
    return true;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string name = ms_context->backend_policy();
#ifdef WITH_BACKEND
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    if (!device_context->GetDeprecatedInterface()->IsTsdOpened(ms_context)) {
      InitPipeline();
    }
  }
#endif

  if (name == kMsConvert || name == kMsVm || name == "ge") {
#ifdef WITH_BACKEND
    if (iter_num == -1) {
      iter_num = INT32_MAX;
    }
    bool status = InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
    return status;
#endif
  }
  return name == "ge" ? true : false;
}

bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_ps_mode() && ps::PSContext::instance()->cache_enable() &&
      !ps::PSContext::instance()->is_worker()) {
    return true;
  }
#endif
  PROF_START(InitExecDatasetVm);
  MS_LOG(INFO) << "Start InitDataSet Entry";
  mindspore::python_adapter::set_python_env_flag(true);
  ShapeVector int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int64_t>(item); });
  std::vector<ShapeVector> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t> &item) {
                         ShapeVector vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int64_t>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int64_t>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int64_t>(batch_size)));
  p_init->set_attr("types", MakeValue(types));
  p_init->set_attr("shapes", MakeValue(int_shapes));
  p_init->set_attr("input_indexes", MakeValue(int_input_indexes));

  const std::vector<std::string> empty_str_list;
  p_init->set_attr("input_names", MakeValue(empty_str_list));
  p_init->set_attr("output_names", MakeValue(empty_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodeWeakPtrList({NewValueNode(p_init)}), func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);
  // Before the graph compiling, need reset the iter num.
  ConfigManager::GetInstance().ResetIterNum();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif

  compile::SetMindRTEnable();
  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // The data set graph compiling and running of mindRT.
  if (context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
#if defined(__linux__) && defined(WITH_BACKEND)
    if (ps::PSContext::instance()->is_worker() && ps::PSContext::instance()->cache_enable()) {
      distributed::DataQueueManager::GetInstance().CreateDataQueue(queue_name, size, 128);
    }
#endif

    const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
    MS_EXCEPTION_IF_NULL(mindrt_backend);
    SetRunMode(func_graph, mindrt_backend.get());
    auto &actor_info = mindrt_backend->CompileGraphs(func_graph);
    VectorRef args;
    if (need_run) {
      VectorRef outputs;
      mindrt_backend->RunGraph(actor_info, args, &outputs);
    }
    ConfigManager::GetInstance().set_iter_num(queue_name, size);
    return true;
  }

  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  auto segment = std::make_shared<GraphSegment>(std::vector<AnfNodePtr>{app_init}, false);
  auto runner = convert_fn(segment, "");
  ConfigManager::GetInstance().set_iter_num(queue_name, size);

  if (!(*runner.run)) {
    // empty function
    MS_LOG(EXCEPTION) << "Backend " << backend->name() << " unsupported tdt dataset.";
  }

  // launch init dataset runner without inputs and outputs
  VectorRef args;
  auto fn = runner.run;
  if (need_run) {
    (void)(*fn)(args);
  }
  MS_LOG(DEBUG) << "InitDataSetVm End.";
  PROF_END(InitExecDatasetVm);
  return true;
}

std::string GetJitLevel() {
  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    return iter->second;
  }
  return "";
}

void ResetOpId() { mindspore::id_generator::reset_id(); }
void ResetOpIdWithOffset() { mindspore::id_generator::reset_id_with_offset(); }

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void FinalizeHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    FinalizeBackend();
    return;
  }
#endif
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  device::DeviceContextManager::GetInstance().UnloadPlugin();
}

uint32_t GetHcclRankId() {
  uint32_t rank_id = 0;
  bool ret = CommManager::GetInstance().GetRankID("", &rank_id);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank id failed, return rank id " << rank_id << " as default.";
  }
  return rank_id;
}

uint32_t GetHcclRankSize() {
  uint32_t rank_size = 0;
  bool ret = CommManager::GetInstance().GetRankSize("", &rank_size);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank size failed, return rank size " << rank_size << " as default.";
  }
  return rank_size;
}

void GraphExecutorPy::ExportGraph(const std::string &file_name, const std::string &phase, const py::object encrypt,
                                  char *key) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target != kAscendDevice) {
    MS_EXCEPTION(ValueError) << "Only support export file in 'AIR' format with Ascend backend.";
  }

  bool is_save_to_file = true;
  if (key != nullptr) {
    if (py::isinstance<py::none()>(encrypt)) {
      MS_LOG(ERROR) << "ERROR: encrypt is not a function";
      return;
    }
    is_save_to_file = false;
  }
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(ERROR) << "Phase " << phase << " must compile.";
    return;
  }
  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
  MS_EXCEPTION_IF_NULL(mindrt_backend);
  FuncGraphPtr func_graph = info_[phase]->func_graph;
  MS_EXCEPTION_IF_NULL(func_graph);

  string save_str = mindrt_backend->ExportDFGraph(file_name, func_graph, is_save_to_file);
  if (is_save_to_file) {
    return;
  }
  // save_to_mem in GE & save to file use encrypt
  py::bytes model_bytes(save_str);
  py::bytes key_bytes(key);

  // call python encrypt func
  py::bytes encrypted_model_stream = encrypt(model_bytes, key_bytes);
  if (encrypted_model_stream == py::none()) {
    MS_LOG(ERROR) << "ERROR: Model encrypt fail";
    return;
  }
  // save to file
  std::ofstream ofs(file_name);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "ERROR: Open File '" << file_name << "' failed!";
    return;
  }
  ofs << std::string(encrypted_model_stream);
  ofs.close();
}

FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                        const std::string &dec_mode, const py::object decrypt) {
  FuncGraphPtr func_graph = nullptr;
  if (dec_mode == "Customized") {
    py::bytes key_bytes(dec_key);
    py::bytes model_stream = decrypt(file_name, key_bytes);
    std::string model_string(model_stream);

    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  } else {
    MindIRLoader mindir_loader(false, reinterpret_cast<const unsigned char *>(dec_key), key_len, dec_mode, false);
    func_graph = mindir_loader.LoadMindIR(file_name);
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  return func_graph;
}

FuncGraphPtr SplitMindIR(const std::string &file_name) {
  MS_LOG(INFO) << "Start split mindir";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  FuncGraphManagerPtr func_graph_manager = func_graph->manager();

  MS_LOG(INFO) << "func_graph_manager is not null";
  if (func_graph_manager == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
    func_graph_manager->AddFuncGraph(func_graph);
  }
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_manager(func_graph_manager);

  // Get the parameters items and add the value to args_abs.
  auto params = func_graph->parameters();
  auto inputs = func_graph->get_inputs();
  for (std::size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i]->abstract();
    (void)parallel::ExtendInputArgsAbstractShape(input, i);
  }
  parallel::StepAutoParallel(func_graph, NULL);
  parallel::StepParallel(func_graph, NULL);
  parallel::StepAllreduceFusion(func_graph, NULL);
  resource->set_func_graph(func_graph);
  resource->set_manager(func_graph->manager());
  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig virtual_dataset = opt::OptPassConfig({irpass.virtual_dataset_eliminate_});
  opt::OptPassConfig virtual_output = opt::OptPassConfig({irpass.virtual_output_eliminate_});

  opt::OptPassGroupMap map_parallel_eliminate(
    {{"virtual_dataset", virtual_dataset}, {"virtual_output", virtual_output}});

  auto split_pass_opts = opt::Optimizer::MakeOptimizer("map_parallel_eliminate", resource, map_parallel_eliminate);
  ProfileExecute(MsProfile::GetProfile()->Step("split_pass_opts"),
                 [&split_pass_opts, &func_graph]() { func_graph = split_pass_opts->step(func_graph, true); });

  AbstractBasePtrList args_abs_list;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                       [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
  func_graph = pipeline::Renormalize(resource, func_graph, args_abs_list);

  resource->set_args_abs(args_abs_list);

  MindIRExporter mindir_exporter;
  mindir_exporter.ExportProto(func_graph, "split_net", nullptr);

  parallel::HandleGroupInfo();

  return func_graph;
}

FuncGraphPtr SplitDynamicMindIR(const std::string &file_name, size_t device_num, size_t rank_id, bool sapp) {
  MS_LOG(INFO) << "Start split dynamic mindir for transformer network";
  FuncGraphPtr func_graph = nullptr;
  MindIRLoader mindir_loader;
  func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file.";
    return nullptr;
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("load.ir", func_graph);
  }
#endif
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  parallel_context->Reset();
  parallel_context->set_parallel_mode(parallel::kAutoParallel);
  parallel_context->set_strategy_search_mode(parallel::kRecursiveProgramming);
  parallel_context->set_direct_split(true);
  parallel_context->set_full_batch(true);
  parallel_context->set_group_ckpt_save_file("group_info");

  for (size_t rank_id_iter = 0; rank_id_iter < device_num; rank_id_iter++) {
    auto tmp_func_graph = mindspore::BasicClone(func_graph);
    FuncGraphManagerPtr func_graph_manager = tmp_func_graph->manager();

    if (func_graph_manager == nullptr) {
      MS_LOG(INFO) << "func_graph_manager is null";
      std::vector<FuncGraphPtr> graphs{tmp_func_graph};
      func_graph_manager = std::make_shared<FuncGraphManager>(graphs);
      func_graph_manager->AddFuncGraph(tmp_func_graph);
    }

    auto inputs = tmp_func_graph->get_inputs();
    for (std::size_t i = 0; i < inputs.size(); i++) {
      auto input = inputs[i]->abstract();
      (void)parallel::ExtendInputArgsAbstractShape(input, i);
    }

    auto res = parallel::StepAssignedParallel(tmp_func_graph, func_graph_manager, device_num, rank_id_iter, sapp);
    if (!res) {
      MS_LOG(ERROR) << "StepAssignedParallel failed. Please check.";
      return nullptr;
    }
    pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
    resource->set_is_load(false);
    resource->set_manager(func_graph_manager);
    resource->set_func_graph(tmp_func_graph);
    // Get the parameters items and add the value to args_abs.
    auto params = tmp_func_graph->parameters();
    AbstractBasePtrList args_abs_list;
    (void)std::transform(params.begin(), params.end(), std::back_inserter(args_abs_list),
                         [](const AnfNodePtr &p) -> AbstractBasePtr { return p->abstract(); });
    tmp_func_graph = pipeline::Renormalize(resource, tmp_func_graph, args_abs_list);

#ifdef ENABLE_DUMP_IR
    auto re_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(re_context);
    if (re_context->CanDump(kIntroductory)) {
      string renormalize_net_name = "Renomalize_" + std::to_string(rank_id_iter) + ".ir";
      DumpIR(renormalize_net_name, tmp_func_graph);
    }
#endif

    parallel::HandleGroupInfo();
    string net_save_name = "split_net" + std::to_string(rank_id_iter);
    MindIRExporter mindir_exporter;
    res = mindir_exporter.ExportProto(tmp_func_graph, net_save_name, nullptr);
    if (!res) {
      MS_LOG(ERROR) << "Export MindIR file failed failed. Please check.";
      return nullptr;
    }
  }

  return func_graph;
}

void CloseTsd(bool force) {
#ifdef WITH_BACKEND
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    (void)device_context->GetDeprecatedInterface()->CloseTsd(context_ptr, force);
  }
#endif
}

void InitPipeline() {
  // set python env flag
  RecordInitStatus();
  mindspore::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  CompileConfigManager::GetInstance().CollectCompileConfig();
}

void FinalizeBackend() { CloseTsd(); }

void MemoryRecycle() {
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif
  ReclaimOptimizer();
  session::ExecutorManager::Instance().ClearDoneTasks();
  ad::g_k_prims.clear();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
  kArgsCache.clear();
  kCellArgsMap.clear();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  ConfigManager::GetInstance().ResetConfig();
  ScopeManager::GetInstance().ClearScope();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  FuncGraphLoopBreaker::Inst().BreakLoop();
}

void BindDeviceCtx() { device::DeviceContextManager::GetInstance().BindDeviceCtx(); }

void ClearResPart1() {
  pynative::PyNativeExecutor::GetInstance()->WorkerJoin();
  runtime::OpExecutor::GetInstance().WorkerJoin();
  // When the python process exits, the kernels on the device may not have finished executing.
  device::KernelRuntimeManager::Instance().WaitTaskFinishOnDevice();
  device::DeviceContextManager::GetInstance().WaitTaskFinishOnDevice();
  tensor::StubTensorConverter::GetInstance().Clear();
  RecordExitStatus();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
  mindspore::RDR::ResetRecorder();
#endif
  runtime::GraphScheduler::GetInstance().Clear();
  runtime::ProfilerAnalyzer::GetInstance().Clear();
  opt::PassConfigure::Instance().Clear();

  MS_LOG(INFO) << "Start Finalize StreamSynchronizer...";
  device::StreamSynchronizer::GetInstance()->Finalize();
  MS_LOG(INFO) << "End Finalize StreamSynchronizer...";

  PrimitivePy::ClearHookRes();
  ad::g_k_prims.clear();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();

  abstract::ClearPrimEvaluatorMap();
  pipeline::GetMethodMap().clear();
  pipeline::GetAttrMap().clear();
#ifdef WITH_BACKEND
  pipeline::GraphExecutorPy::GetInstance()->ClearInfo();
  pipeline::JitExecutorPy::GetInstance()->ClearInfo();
#endif
  pipeline::GraphExecutorPy::ClearRes();
  pipeline::JitExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
}

void ClearResPart2() {
  MS_LOG(INFO) << "Start clear PyNativeExecutor...";
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  MS_LOG(INFO) << "End clear PyNativeExecutor.";

  MS_LOG(INFO) << "Start clear ConfigManager...";
  ConfigManager::GetInstance().ResetIterNum();
  MS_LOG(INFO) << "End clear ConfigManager.";

  session::ExecutorManager::Instance().Clear();

  MS_LOG(INFO) << "Start clear device context...";
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  MS_LOG(INFO) << "End clear device context.";

  MS_LOG(INFO) << "Start clear kernel runtime...";
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  MS_LOG(INFO) << "End clear kernel runtime.";

  MS_LOG(INFO) << "Start clear CollectiveManager...";
  // for GE, HcclCommDestroy should after RemoveGraph in ClearGraphWrapper in ClearDeviceContexts
  (void)distributed::collective::CollectiveManager::instance()->Finalize();
  MS_LOG(INFO) << "End clear CollectiveManager.";

  MS_LOG(INFO) << "Start clear AnalysisResultCacheMgr...";
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  MS_LOG(INFO) << "End clear AnalysisResultCacheMgr.";

  MS_LOG(INFO) << "Start clear AnalysisContext...";
  abstract::AnalysisContext::ClearContext();
  MS_LOG(INFO) << "End clear AnalysisContext...";

  MS_LOG(INFO) << "Start clear AnalysisSchedule...";
  abstract::AnalysisSchedule::GetInstance().Stop();
  MS_LOG(INFO) << "End clear AnalysisSchedule...";
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  debugger->Reset();
#endif
  kArgsCache.clear();
  kCellArgsMap.clear();
}

void ClearResPart3() {
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  MS_LOG(INFO) << "Start clear ClearObjectCache...";
  parse::data_converter::ClearObjectCache();
  MS_LOG(INFO) << "End clear ClearObjectCache...";

  MS_LOG(INFO) << "Start clear Parser...";
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "End clear Parser...";

  MS_LOG(INFO) << "Start ClearTraceStack...";
  trace::ClearTraceStack();
  MS_LOG(INFO) << "End ClearTraceStack...";

  MS_LOG(INFO) << "Start clear InterpretNodeRecorder...";
  InterpretNodeRecorder::GetInstance().Clear();
  MS_LOG(INFO) << "End clear InterpretNodeRecorder...";

  MS_LOG(INFO) << "Start clear parallel::entire_costgraph...";
  parallel::entire_costgraph.reset();
  MS_LOG(INFO) << "End clear parallel::entire_costgraph...";

  MS_LOG(INFO) << "Start clear ProtobufLibrary...";
  google::protobuf::ShutdownProtobufLibrary();
  MS_LOG(INFO) << "End clear ProtobufLibrary...";

  MS_LOG(INFO) << "Start clear ParserDefaultObjects ...";
  pynative::ParserDefaultObjects::GetInstance().ClearRes();
  MS_LOG(INFO) << "End clear ParserDefaultObjects...";

  // ResetPythonScope after all py::object is freed.
  MS_LOG(INFO) << "Start clear python_adapter...";
  python_adapter::ResetPythonScope();
  MS_LOG(INFO) << "End clear python_adapter.";
}

void ClearSingleton() {
  MS_LOG(INFO) << "Start clear singleton...";
  profiler::Profiler::Clear();
  debug::tft::TFTWaitSem::GetInstance().Clear();
#ifdef ENABLE_AKG
  kernel::GraphKernelBuildManager::Instance().Clear();
#endif
  somas::SomasManager::Instance().Clear();
  GraphKernelInfoManager::Instance().Clear();
  device::DataQueueMgr::GetInstance().Clear();
  session::SessionFactory::Get().Clear();
  device::KernelRuntimeManager::Instance().Clear();
  ExecuteOrderTracker::GetInstance().Clear();
  OpPrimPyRegister::GetInstance().Clear();
  DumpJsonParser::Finalize();
  AclDumpJsonWriter::Finalize();
  CommManager::Clear();
  expander::ClearAllCache();

  MS_LOG(INFO) << "End clear singleton.";
}

void ClearResAtexit() {
  MS_LOG(INFO) << "Pipeline clear all resource";
  try {
    MsException::Instance().CheckException();
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "Check exception before process exit: " << e.what();
  }
  ClearResPart1();
  ClearResPart2();

  mindspore::trans::FormatHelper::GetInstance().Clear();
  ClearResPart3();
  ClearSingleton();
  //  The premature unloading of the plugin .so triggers the process to exit during the termination phase. Other
  //  components' singletons, static variables, and global variables in MindSpore may inadvertently invoke the plugin
  //  interface, resulting in an undefined coredump.
}

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode) {
  size_t encrypt_len;
  auto encrypt_data = mindspore::Encrypt(&encrypt_len, reinterpret_cast<Byte *>(plain_data), plain_len,
                                         reinterpret_cast<Byte *>(key), key_len, enc_mode);
  if (encrypt_data == nullptr) {
    MS_EXCEPTION(ValueError) << "Encrypt failed";
  }
  auto py_encrypt_data = py::bytes(reinterpret_cast<char *>(encrypt_data.get()), encrypt_len);
  return py_encrypt_data;
}

py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data =
    mindspore::Decrypt(&decrypt_len, encrypt_data_path, reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

py::bytes PyDecryptData(char *model_data, size_t data_size, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data = mindspore::Decrypt(&decrypt_len, reinterpret_cast<Byte *>(model_data), data_size,
                                         reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

bool PyIsCipherFile(const std::string &file_path) { return mindspore::IsCipherFile(file_path); }

void FinalizeCluster() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    if (!distributed::cluster_exit_with_exception()) {
      MS_LOG(INFO) << "Start finalize the cluster instance.";
      // Finalize MindSpore cluster only when this process exits without any exception.
      (void)distributed::cluster::ClusterContext::instance()->Finalize(UINT32_MAX);
      MS_LOG(INFO) << "End finalize the cluster instance.";
    } else {
      (void)distributed::cluster::ClusterContext::instance()->StopThreadsOnException();
    }
  }
#endif
}

void SwapCache(const tensor::TensorPtr &host, const tensor::TensorPtr &device, const tensor::TensorPtr &block_mapping,
               const bool &is_device_to_host) {
  auto block_mapping_shape = block_mapping->shape();
  const size_t num_two = 2;
  if (block_mapping_shape.size() != num_two) {
    MS_LOG_EXCEPTION << "The shape size of Cache input mapping tensor should be 2, but got: "
                     << block_mapping_shape.size();
  }
  if (block_mapping_shape[kIndex1] != num_two) {
    MS_LOG_EXCEPTION << "The second dim of CacheKernel input mapping tensor should be 2, but got: "
                     << block_mapping_shape[0];
  }

  auto in_shape = device->shape();
  auto type_byte = GetTypeByte(TypeIdToType(host->data_type()));
  size_t block_size_in_bytes = LongToSize(
    std::accumulate(in_shape.begin() + kIndex1, in_shape.end(), SizeToLong(type_byte), std::multiplies<int64_t>()));

  uint8_t *host_ptr = reinterpret_cast<uint8_t *>(host->data_c());
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(device->device_address());
  MS_EXCEPTION_IF_NULL(device_addr);
  uint8_t *device_ptr = reinterpret_cast<uint8_t *>(const_cast<void *>(device_addr->GetPtr()));
  MS_EXCEPTION_IF_NULL(device_ptr);

  auto block_mapping_data = reinterpret_cast<int64_t *>(block_mapping->data_c());
  for (size_t i = 0; i < LongToSize(block_mapping_shape[0]); i++) {
    int64_t src_block_num = block_mapping_data[num_two * i];
    int64_t dst_block_num = block_mapping_data[num_two * i + kIndex1];
    size_t src_block_offset = LongToSize(src_block_num) * block_size_in_bytes;
    size_t dst_block_offset = LongToSize(dst_block_num) * block_size_in_bytes;

    if (is_device_to_host) {
      device_addr->CopyDeviceToHost(host_ptr + dst_block_offset, device_ptr + src_block_offset, block_size_in_bytes);
    } else {
      device_addr->CopyHostToDevice(device_ptr + dst_block_offset, host_ptr + src_block_offset, block_size_in_bytes);
    }
  }
}

py::object BaseRefToPyDataWithUserData(const BaseRef &value, const AbstractBasePtr &abs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kGraphExecutorPy, runtime::ProfilerEvent::kOutputProcess,
                                     "BaseRefToPyData");
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  if (!allow_fallback_runtime) {
    return BaseRefToPyData(value, abs);
  }
  if (utils::isa<ValuePtr>(value)) {
    // Do not use abs as input to BaseRefToPyData, since the res need to be a tensor to get user data.
    auto res = BaseRefToPyData(value);
    MS_LOG(DEBUG) << "res: " << py::str(res);
    const auto user_data = GetUserDataFromAddress(res);
    if (user_data != nullptr) {
      return user_data->obj;
    } else {
      MS_LOG(DEBUG) << "user data is empty";
    }
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    return GetVectorRefPyData(vec_ref, abs);
  }
  return BaseRefToPyData(value, abs);
}

void PreJit(const py::object &args, const py::object &kwargs) {
  const auto &self = GetSelfFromArgs(args);
  parse::Parser::InitParserEnvironment(self);
}
}  // namespace pipeline
}  // namespace mindspore
