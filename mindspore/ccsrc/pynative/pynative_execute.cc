/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "pynative/pynative_execute.h"
#include "availability/silent_check/silent_check.h"
#include "pynative/pynative_utils.h"
#include "pynative/grad/function_py.h"
#include "pynative/predict_out_type_map.h"
#include "pynative/op_function/auto_grad_register.h"
#include "include/common/utils/tensor_py.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pybind_api/pybind_patch.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/grad/hook_py.h"
#include "include/common/utils/config_manager.h"
#include "include/common/pybind_api/api_register.h"
#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/ps/pass.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/lazy_fusion.h"
#include "debug/profiler/profiler.h"
#include "debug/profiler/profiling.h"
#include "ir/cell.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/utils/python_utils.h"
#include "common/kernel_mod_cache.h"
#include "runtime/pipeline/pipeline.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/pynative/adapter.h"
#include "include/common/pynative/variable.h"

namespace mindspore::pynative {
std::shared_ptr<PyNativeExecutor> PyNativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PyNativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PyNativeExecutor::grad_executor_ = nullptr;
std::mutex PyNativeExecutor::instance_lock_;
namespace {
template <typename T, typename... Args>
T PyNativeExecutorTry(const std::function<T(const Args &...)> &method, const Args &... args) {
  const auto &inst = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  MS_EXCEPTION_IF_NULL(method);
  auto already_set_error_handler = [&inst]() {
    // Print function call stack info before release.
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // Call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info.
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    inst->ClearRes();
  };

  if constexpr (std::is_same_v<T, void>) {
    HandleExceptionRethrow([&method, &args...]() { method(args...); }, already_set_error_handler,
                           [&inst]() { inst->ClearRes(); }, [&inst]() { inst->ClearRes(); });
  } else {
    T res;
    HandleExceptionRethrow([&res, &method, &args...]() { res = method(args...); }, already_set_error_handler,
                           [&inst]() { inst->ClearRes(); }, [&inst]() { inst->ClearRes(); });
    return res;
  }
}

// Tensor may be used before the execution of the asynchronous task.
void SetCallbackForInputTensor(const std::vector<ValuePtr> &input_values) {
  for (auto &input : input_values) {
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<tensor::Tensor>()) {
      auto tensor = input->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->set_need_pipeline_sync(true);
    }
  }
}
}  // namespace

void PyNativeExecutor::StoreAsyncStatus(const PyboostOpRunInfoPtr &op_run_info) const {
  // Pure function running or cell not set mix precision
  op_run_info->async_status.disable_mix_precision = forward_executor()->CellNotSetMixedPrecision(op_run_info);
  op_run_info->async_status.is_jit_compiling = forward_executor()->is_jit_compiling();
  op_run_info->async_status.custom_bprop_cell_count = grad_executor()->custom_bprop_cell_count();
}

void PyNativeExecutor::StoreAsyncStatus(const FrontendOpRunInfoPtr &op_run_info) const {
  // Pure function running or cell not set mix precision
  op_run_info->async_status.disable_mix_precision = forward_executor()->CellNotSetMixedPrecision(op_run_info);
  op_run_info->async_status.is_jit_compiling = forward_executor()->is_jit_compiling();
  op_run_info->async_status.custom_bprop_cell_count = grad_executor()->custom_bprop_cell_count();
}

py::object PyNativeExecutor::RunOpStub(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args, true);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     op_run_info->base_op_run_info.op_name, false, true);
  SetCallbackForInputTensor(op_run_info->op_grad_info->input_value);

  StoreAsyncStatus(op_run_info);
  forward_executor()->WaitForwardTask();
  // RunOp sync
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  return py::reinterpret_steal<py::object>(tensor::Wrap(op_run_info->real_out));
}

py::object PyNativeExecutor::RunSliceOpStub(const std::vector<ValuePtr> &input_values,
                                            const std::vector<SliceOpInfoPtr> &slice_op_infos) const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunOp);

  forward_executor()->Init();
  auto stream_id = forward_executor()->GetStreamId();
  SetCallbackForInputTensor(input_values);
  auto requires_grad = GradState::Get().RequiresGrad();
  forward_executor()->WaitForwardTask();
  auto ret = forward_executor()->RunSliceOpFrontend(input_values, slice_op_infos, requires_grad, nullptr, stream_id);
  return py::reinterpret_steal<py::object>(tensor::Wrap(ret));
}

void PyNativeExecutor::SetCreationType(const py::object &obj, autograd::CreationType creation_type) {
  forward_executor()->WaitForwardTask();
  if (!tensor::IsTensorPy(obj)) {
    MS_LOG(EXCEPTION) << "Input obj is not a Tensor";
  }
  auto tensor = tensor::ConvertToTensor(obj);
  auto view_autograd_meta_data = autograd::impl::GetViewAutogradMetaImpl(tensor);
  if (!view_autograd_meta_data) {
    MS_LOG(EXCEPTION) << "Tensor has no ViewAutogradMeta";
  }
  view_autograd_meta_data->set_creation_type(creation_type);
}

py::object PyNativeExecutor::RealRunOp(const py::args &args) const {
  FrontendOpRunInfoPtr op_run_info = forward_executor()->GenerateOpRunInfo(args);
  StoreAsyncStatus(op_run_info);
  PyNativeExecutorTry(forward_executor()->RunOpS, op_run_info);
  if (PyGILState_Check() == 0) {
    py::gil_scoped_acquire acquire;
    return py::reinterpret_steal<py::object>(tensor::Wrap(op_run_info->real_out));
  }
  return py::reinterpret_steal<py::object>(tensor::Wrap(op_run_info->real_out));
}

py::object PyNativeExecutor::CallConstantFolding(const py::args &args) const {
  return forward_executor()->infer_operation()->CallConstantFolding(args);
}

void PyNativeExecutor::set_py_exe_path(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  const auto &py_exe_path_s = py_exe_path.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void PyNativeExecutor::set_kernel_build_server_dir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  const auto &kernel_build_server_dir_s = kernel_build_server_dir.cast<std::string>();
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

void PyNativeExecutor::ClearRes() const {
  runtime::Pipeline::Get().WaitAll();
  silentcheck::SilentCheckerBase::ClearAll();
  // Clear forward tasks before clear op graphs cache.
  pynative::OpCompiler::GetInstance().ClearAllCache();
  kernel::KernelModCache::GetInstance().ClearAllCache();
  grad_executor()->jit()->ClearAutoGradCache();
  PyNativeAlgo::Common::ClearRes();
  autograd::RegisterHook::ClearHookMap();

  // Maybe exit in runop step
  auto ms_context = MsContext::GetInstance();
  if (ms_context != nullptr) {
    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  }
  ConfigManager::GetInstance().ResetIterNum();
  if (forward_executor_ != nullptr) {
    forward_executor_->ClearRes();
  }
  if (grad_executor_ != nullptr) {
    grad_executor_->ClearRes();
  }
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
  MS_LOG(DEBUG) << "Clear all res";
}

void PyNativeExecutor::Init() {
  MS_LOG(DEBUG) << "Init PyNativeExecutor";
  forward_executor_ = std::make_shared<ForwardExecutor>();
  forward_executor_->Init();
  grad_executor_ = std::make_shared<GradExecutor>(forward_executor_);
  grad_executor_->Init();
  forward_executor_->set_grad_executor(grad_executor_);
  forward_executor_->RefreshForwardCallback();
  runtime::ProfilerAnalyzer::GetInstance().SetThreadIdToName(std::this_thread::get_id(), "Python");
  LazyFusionInit();
  OpsAutoGradImplRegister();
}

void PyNativeExecutor::Sync() const { PyNativeExecutorTry(forward_executor()->SyncData, true); }

bool PyNativeExecutor::grad_flag() const { return GradState::Get().grad_flag(); }

void PyNativeExecutor::set_grad_flag(bool flag) const { GradState::Get().set_grad_flag(flag); }

bool PyNativeExecutor::enable_grad() const { return GradState::Get().enable_grad(); }

void PyNativeExecutor::set_enable_grad(bool enable_grad) const { GradState::Get().set_enable_grad(enable_grad); }

bool PyNativeExecutor::RequiresGrad() const { return GradState::Get().RequiresGrad(); }

bool PyNativeExecutor::IsHighOrder() const { return grad_executor()->IsHighOrderTopCell(); }

py::object PyNativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &obj,
                                             const py::object &weights, const py::object &grad_hash_id,
                                             const py::args &args, const py::kwargs &kwargs) const {
  return grad_executor()->CheckAlreadyRun(grad, obj, weights, grad_hash_id, args, kwargs);
}

void PyNativeExecutor::NewGraph(const py::object &obj, const py::args &args) const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeNewGraph,
                                     runtime::ProfilerRecorder::kNoName, false);
  PyNativeExecutorTry(grad_executor()->InitGraph, obj, args);
}

void PyNativeExecutor::EndGraph(const py::object &obj, const py::object &out, const py::args &args) const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeEndGraph,
                                     runtime::ProfilerRecorder::kNoName, false);
  PyNativeExecutorTry(grad_executor()->LinkGraph, obj, out, args);
}

py::object PyNativeExecutor::RunGrad(const prim::GradOperationPtr &grad, const py::object &cell,
                                     const py::object &weights, const py::object &grad_position,
                                     const py::object &has_aux, const py::args &args) const {
  runtime::ProfilerStageRecorder recorder(runtime::ProfilerStage::kRunGrad);
  return PyNativeExecutorTry(grad_executor()->Run, grad, cell, weights, grad_position, has_aux, args);
}

py::object PyNativeExecutor::GradJit(const py::args &args) const {
  return PyNativeExecutorTry(grad_executor()->GradJit, args);
}

py::object PyNativeExecutor::CallCustomBprop(const py::object &cell_obj, const py::object &out,
                                             const py::args &args) const {
  MS_EXCEPTION_IF_NULL(grad_executor()->top_cell());
  return PyNativeExecutorTry(grad_executor()->CallCustomBpropFunc, cell_obj, out, args);
}

void PyNativeExecutor::SetMixedPrecisionType(const MixedPrecisionType mix_type, bool is_push) const {
  return forward_executor()->set_mix_precision_type(mix_type, is_push);
}

void PyNativeExecutor::WorkerJoin() {
  GilReleaseWithCheck release_gil;
  runtime::Pipeline::Get().frontend_stage()->WorkerJoin();
  runtime::Pipeline::Get().stress_detect()->WorkerJoin();
}

void PyNativeExecutor::SetJitCompileStatus(bool is_compiling, const std::string &phase) const {
  forward_executor()->set_is_jit_compiling(is_compiling);
  grad_executor()->jit()->set_graph_phase(phase);
}

void PyNativeExecutor::SetIsRunRecompute(bool is_runing_recompute) const {
  grad_executor()->set_is_run_recompute(is_runing_recompute);
}

void PyNativeExecutor::set_forward_use_dynamic_shape_process(bool flag) const {
  grad_executor()->set_forward_use_dynamic_shape_process(flag);
  if (flag) {
    profiler::ProfilerManager::GetInstance()->SetNetDynamicShapeStatus();
  }
}

void PyNativeExecutor::SetDynamicInput(const py::object &obj, const py::args &args) const {
  grad_executor()->SaveDynamicInputsCells(obj, args);
}

py::object PyNativeExecutor::GetDynamicInput(const py::object &actual_input) const { return actual_input; }

void PyNativeExecutor::ParentBeforeFork() {
  MS_LOG(DEBUG) << "PyNativeExecutor prepare before fork.";
  runtime::Pipeline::Get().ParentBeforeFork();
  MS_LOG(DEBUG) << "PyNativeExecutor prepare before fork done.";
}

void PyNativeExecutor::ChildAfterFork() {
  MS_LOG(DEBUG) << "PyNativeExecutor reinitialize after fork.";
  MS_LOG(DEBUG) << "Clear OpCompiler Cache.";
  runtime::Pipeline::Get().ChildAfterFork();
  pynative::OpCompiler::GetInstance().ClearAllCache();
  if (forward_executor_ != nullptr) {
    MS_LOG(DEBUG) << "Clear forward_executor_ resources.";
    forward_executor_->ClearRes();
    // Call ForwardExecutor::ReInit() to update device_target_
    forward_executor_->ReInit();
    MS_LOG(DEBUG) << "Reinitialize forward_executor_.";
    forward_executor_->ChildAfterFork();
  }
  // Reset PyNativeExecutor resources
  if (grad_executor_ != nullptr) {
    MS_LOG(DEBUG) << "Clear grad_executor_ resources.";
    grad_executor_->ClearRes();
    MS_LOG(DEBUG) << "Reinitialize grad_executor_.";
    grad_executor_->ChildAfterFork();
  }
  runtime::OpRunner::ChildAfterFork();
  OpsAutoGradImplRegister();
  MS_LOG(DEBUG) << "PyNativeExecutor reinitialize after fork done.";
}

void PyNativeExecutor::SetAsyncForGraph(bool flag) const {
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(INFO) << "Skip set async flag " << flag;
    return;
  }
  if (!flag) {
    // Need to wait all tasks finish before disable async.
    runtime::Pipeline::Get().WaitAll();
  }
  MS_LOG(INFO) << "Set async " << flag << " for GRAPH_MODE";
  runtime::Pipeline::Get().SetSpin(flag);
  runtime::OpExecutor::GetInstance().set_async_for_graph(flag);
}

void RegPyNativeExecutor(const py::module *m) {
  autograd::RegFunctionBase(m);

  using autograd::CreationType;
  py::enum_<CreationType>(*m, "CreationType")
    .value("DEFAULT", CreationType::kDefault)
    .value("NO_GRAD_MODE", CreationType::kNoGradMode)
    .value("MULTI_OUTPUT", CreationType::kMultiOutput)
    .value("CUSTOM_BPROP", CreationType::kCustomBprop);

  (void)py::class_<PyNativeExecutor, std::shared_ptr<PyNativeExecutor>>(*m, "PyNativeExecutor_")
    .def_static("get_instance", &PyNativeExecutor::GetInstance, "PyNativeExecutor get_instance.")
    .def("set_mixed_precision_type", &PyNativeExecutor::SetMixedPrecisionType, "set cell mixed precision type.")
    .def("new_graph", &PyNativeExecutor::NewGraph, "pynative new a graph.")
    .def("end_graph", &PyNativeExecutor::EndGraph, "pynative end a graph.")
    .def("check_run", &PyNativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
    .def("grad_jit", &PyNativeExecutor::GradJit, "pynative grad for jit.")
    .def("call_custom_bprop", &PyNativeExecutor::CallCustomBprop, "pynative custom bprop")
    .def("clear_res", &PyNativeExecutor::ClearRes, "pynative clear exception res.")
    .def("sync", &PyNativeExecutor::Sync, "pynative sync stream.")
    .def("grad", &PyNativeExecutor::RunGrad, "pynative executor run grad.")
    .def("grad_flag", &PyNativeExecutor::grad_flag, "pynative grad flag")
    .def("enable_grad", &PyNativeExecutor::enable_grad, "pynative enable grad, used for with no_grad")
    .def("requires_grad", &PyNativeExecutor::RequiresGrad, "Is current need grad")
    .def("set_grad_flag", &PyNativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
         "Executor set grad flag.")
    .def("set_enable_grad", &PyNativeExecutor::set_enable_grad, py::arg("enable_grad") = py::bool_(true),
         "pynative set enable grad")
    .def("high_order", &PyNativeExecutor::IsHighOrder, "pynative high order")
    .def("set_cell_use_dynamic_shape_process", &PyNativeExecutor::set_forward_use_dynamic_shape_process,
         "set eval use dynamic shape process.")
    .def("set_dynamic_input", &PyNativeExecutor::SetDynamicInput, "set dynamic input")
    .def("get_dynamic_input", &PyNativeExecutor::GetDynamicInput, "get dynamic input")
    .def("set_py_exe_path", &PyNativeExecutor::set_py_exe_path, py::arg("py_exe_path") = py::str(""),
         "set python executable path.")
    .def("set_kernel_build_server_dir", &PyNativeExecutor::set_kernel_build_server_dir,
         py::arg("kernel_build_server_dir") = py::str(""), "set kernel build server directory path.")
    .def("set_jit_compile_status", &PyNativeExecutor::SetJitCompileStatus, "set jit compile status.")
    .def("set_is_run_recompute", &PyNativeExecutor::SetIsRunRecompute, "set grad is in recompile status.")
    .def("real_run_op", &PyNativeExecutor::RealRunOp, "run op synchronously")
    .def("run_op_async", &PyNativeExecutor::RunOpStub, "run op asynchronously")
    .def("set_async_for_graph", &PyNativeExecutor::SetAsyncForGraph, py::arg("flag") = py::bool_(false),
         "Executor set async flag.")
    .def("constant_folding", &PyNativeExecutor::CallConstantFolding, "Call Constant Folding Primitive")
    .def("set_creation_type", &PyNativeExecutor::SetCreationType, "Set tensor's view creation type");
}

struct PyNativeExecutorRegister {
  PyNativeExecutorRegister() {
    PyNativeAdapter::SetGradJitHandler(
      [](const py::args &args) -> py::object { return PyNativeExecutor::GetInstance()->GradJit(args); });
    PyNativeAdapter::SetSetGraphPhaseHandler([](const std::string &graph_phase) -> void {
      PyNativeExecutor::GetInstance()->grad_executor()->jit()->set_graph_phase(graph_phase);
    });
  }
} pynative_executor_register;
}  // namespace mindspore::pynative
