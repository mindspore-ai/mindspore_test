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

#include "plugin/device/ascend/hal/hardware/ge_device_context.h"
#include <tuple>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include "include/transform/graph_ir/types.h"
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "plugin/device/ascend/device_context_conf/op_debug_conf.h"
#include "plugin/device/ascend/device_context_conf/op_precision_conf.h"
#include "plugin/device/ascend/device_context_conf/op_tuning_conf.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "include/backend/debug/profiler/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"
#include "transform/symbol/acl_base_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_compiler_symbol.h"
#include "kernel/ascend/availability/silent_check/ascend_silent_check.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kOpDebugConfigFile = "ge_op_debug_config.ini";
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";

bool IsNeedHybridMode(const FuncGraphPtr &func_graph) {
  // cell reuse + pipeline parallel
  // only O2
  if (func_graph == nullptr) {
    return false;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  bool has_cell_reuse = std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    if (node == nullptr || !node->isa<CNode>()) {
      return false;
    }
    auto cnode = node->cast<CNodePtr>();
    const auto &inputs = cnode->inputs();

    // for func graph
    AnfNodePtr fn = inputs[0];
    FuncGraphPtr child_graph = common::AnfAlgo::GetValueNodeFuncGraph(fn);
    bool func_graph_has_cell_reuse = child_graph != nullptr && child_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE);

    // for kernel graph
    bool kernel_graph_has_cell_reuse = false;
    if (IsPrimitiveCNode(cnode, prim::kPrimCall)) {
      auto call_graph = cnode->input(kIndex1);
      auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
      kernel_graph_has_cell_reuse = sub_kernel_graph != nullptr && sub_kernel_graph->need_inline();
    }
    return func_graph_has_cell_reuse || kernel_graph_has_cell_reuse;
  });

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_step = parallel_context->grad_accumulation_step();
  MS_LOG(INFO) << "graph: " << func_graph->ToString() << "stages: " << stages << ", grad_accu_step: " << grad_accu_step;
  if (stages <= 1 && grad_accu_step <= 1) {
    if (has_cell_reuse) {
      // no pipeline + cell reuse + O2
      context->SetCellReuseLevel(CellReuseLevel::kNoInline);
    }
    return false;
  }
  if (IsDisableGeKernel()) {
    if (has_cell_reuse) {
      // force subgraph sink
      context->SetCellReuseLevel(CellReuseLevel::kNoInline);
    }
    return false;
  }
  return has_cell_reuse;
}

void SetAclOpDebugOption() {
  auto op_debug_conf = OpDebugConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_debug_conf);
  auto op_debug_option = op_debug_conf->debug_option();
  if (op_debug_option == "oom") {
    auto ret = CALL_ASCEND_API(aclSetCompileopt, aclCompileOpt::ACL_OP_DEBUG_OPTION, op_debug_option.c_str());
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Acl set op debug option: " << op_debug_option << " failed! Error flag is " << ret;
    }
  }
}
}  // namespace

bool GeDeviceContext::PartitionGraph(const FuncGraphPtr &func_graph) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (common::AnfAlgo::IsDynamicShapeFuncGraph(func_graph)) {
    // dynamic shape default kernel be kernel before ge support
    if (GetRunMode(func_graph) == RunMode::kKernelMode) {
      return true;
    }
    opt::GEDynamicUnifyMindIR(func_graph);
    bool all_support = true;
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    const auto &sub_graphs = mng->func_graphs();
    for (const auto &sub_graph : sub_graphs) {
      if (sub_graph == nullptr) {
        continue;
      }
      auto nodes = TopoSort(sub_graph->get_return());
      for (const auto &node : nodes) {
        if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
          continue;
        }
        if (GetCNodeTarget(node) != kAscendDevice) {
          all_support = false;
          continue;
        }
        if (GetCNodePrimitive(node) == nullptr) {
          continue;
        }
        if (!transform::ConvertCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue<std::string>(kCPUDevice), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " can not find adpt, run on CPU";
          continue;
        }
        if (!transform::DynamicShapeSupportCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " not support dynamic shape, will run in KernelGraph";
          continue;
        }
        if (!transform::SinkGraphCheck(node)) {
          all_support = false;
          common::AnfAlgo::SetNodeAttr(kAttrGraphSplitGroup, MakeValue<std::string>(kKernelGroup), node);
          MS_LOG(DEBUG) << node->fullname_with_scope() << " have attrs is not ValueNode, will run in KernelGraph";
        }
      }
    }
    if (!all_support) {
      context_ptr->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    }
  }
  return context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
}

RunMode GeDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (common::AnfAlgo::IsDynamicShapeFuncGraph(func_graph)) {
    if (context->get_param<std::string>(MS_CTX_JIT_LEVEL) == "O2" &&
        context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
      MS_LOG(INFO) << "set dynamic shape RunMode::kGraphMode";
      return RunMode::kGraphMode;
    }
    MS_LOG(INFO) << "dynamic shape default RunMode::kKernelMode";
    // Dynamic shape runs in kbk mode, not support ge graph sink mode.
    auto set_ctx = [&context](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
      context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
      context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
      context->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
    };
    set_ctx(false, false, false);
    return RunMode::kKernelMode;
  }

  if (context->IsKByKExecutorMode() && !context->get_param<bool>(MS_CTX_ENABLE_HYBRID_MODE)) {
    MS_LOG(INFO) << "RunMode::kKernelMode, graph: " << func_graph->ToString();
    return RunMode::kKernelMode;
  } else {
    if (IsNeedHybridMode(func_graph)) {
      context->set_param(MS_CTX_ENABLE_HYBRID_MODE, true);
      MS_LOG(INFO) << "RunMode::kHybridMode, graph: " << func_graph->ToString();
      return RunMode::kHybridMode;
    }
    context->set_param(MS_CTX_ENABLE_HYBRID_MODE, false);
    MS_LOG(INFO) << "RunMode::kGraphMode, graph: " << func_graph->ToString();
    return RunMode::kGraphMode;
  }
}

void GeDeviceContext::Initialize() {
  GilReleaseWithCheck gil_release;
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    return;
  }

  MS_LOG(INFO) << "Start initializing device context.";
  if (UseSimulationApi()) {
    transform::LoadSimulationApiSymbols();
  }

  // set overflow mode
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();
  if (soc_version == "ascend910b" || soc_version == "ascend910_93") {
    bool is_sat = (common::GetEnv("MS_ASCEND_CHECK_OVERFLOW_MODE") == "SATURATION_MODE");
    auto mode = (is_sat) ? aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION
                         : aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_INFNAN;
    auto overflow_mode = (is_sat) ? kSaturationMode : kINFNANMode;
    MS_LOG(INFO) << "The current overflow detection mode is " << overflow_mode << ".";
    auto ret = CALL_ASCEND_API(aclrtSetDeviceSatMode, mode);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Set " << overflow_mode << " mode failed.";
    }
  }

  MS_EXCEPTION_IF_NULL(device_res_manager_);
  device_res_manager_->Initialize();

  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to  heterogeneous mode
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  if (!UseSimulationApi()) {
    graph_executor_->Initialize();
  }

  // should be called after ge initialize.
  SetAclOpDebugOption();

  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
  // DynamicKernelExecutor and KernenlExecutor should be equal for GE
  MS_EXCEPTION_IF_CHECK_FAIL(GetKernelExecutor(true) == GetKernelExecutor(false),
                             "GE dynamic KernelExecutor and KernenlExecutor is not Equal.");
  GetKernelExecutor(false)->Initialize();

  InitDump();
  auto op_tuning_conf = OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  if (op_tuning_conf->EnableAoeOnline()) {
    transform::InitializeAoeUtil();
  }
  if (op_tuning_conf->EnableAoeOffline()) {
    transform::EnableAoeOffline();
  }
  // open tsd
  if (!common::UseDynamicCluster()) {
    if (!GetDeprecatedInterface()->OpenTsd(ms_context)) {
      MS_LOG(EXCEPTION) << "Open tsd failed";
    }
  }
  initialized_ = true;
  pid_ = GetCurrentPID();  // set the pid when first initialize
  MS_LOG(INFO) << "End initializing device context.";
}

void GeDeviceContext::Destroy() {
  if (!IsNeedDestroy()) {
    // The device context is copied from main process by fork
    MS_LOG(INFO) << "The device context is not initialized by current process, it doesn't need to be destroyed.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto op_tuning_conf = OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  if (op_tuning_conf->EnableAoeOnline()) {
    transform::DestroyAoeUtil();
  }
  FinalizeDump();
  if (graph_executor_ == nullptr) {
    return;
  }
  dynamic_cast<GeGraphExecutor *>(graph_executor_.get())->Finalize();
  if (device_res_manager_ == nullptr) {
    return;
  }
  silentcheck::ascend::SilentChecker::GetInstance().ClearCheckHooks();
  // Device resource manager must be destroyed before 'FinalizeGe' unless some runtime APIs will throw exception.
  // for ge, has destropy in graph_executor->finalize
  device_res_manager_->Destroy();

  if (hccl::HcclAdapter::GetInstance().Inited()) {
    (void)hccl::HcclAdapter::GetInstance().FinalizeHccl();
  }
  if (deprecated_interface_ != nullptr) {
    (void)deprecated_interface_->CloseTsd(MsContext::GetInstance(), true);
  }
  initialized_ = false;
}

void GeDeviceContext::InitDump() const {
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (!dump_parser.async_dump_enabled()) {
    return;
  }
  if (dump_parser.FileFormatIsNpy()) {
    if (dump_parser.IsCallbackRegistered()) {
      MS_LOG(INFO) << "DumpDataCallback already registered, no need to register again.";
      return;
    }
    (void)acldumpRegCallback(mindspore::ascend::DumpDataCallBack, 0);
    dump_parser.SetCallbackRegistered();
  }
}

void GeDeviceContext::FinalizeDump() const {
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
  if (!dump_parser.async_dump_enabled()) {
    return;
  }
  if (dump_parser.FileFormatIsNpy() && dump_parser.IsTensorDump()) {
    mindspore::ascend::AscendAsyncDumpManager::GetInstance().WaitForWriteFileFinished();
  }
}

DeprecatedInterface *GeDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>();
  }
  return deprecated_interface_.get();
}

uint32_t GeDeviceContext::GetDeviceCount() {
  uint32_t device_count = 0;
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &device_count);
  if (ret != ACL_ERROR_NONE) {
    MS_EXCEPTION(DeviceProcessError) << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }
  return device_count;
}

std::string GeDeviceContext::GetDeviceName(uint32_t) {
  const char *name = CALL_ASCEND_API(aclrtGetSocName);
  std::string device_name = (name == nullptr) ? "" : name;
  return device_name;
}

uint32_t GeDeviceContext::GetExecuteTimeout() {
  auto op_debug_conf = OpDebugConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_debug_conf);
  return op_debug_conf->execute_timeout();
}

std::string GeDeviceContext::GetAoeJobType() {
  auto op_tuning_conf = OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  return op_tuning_conf->aoe_job_type();
}

std::string GeDeviceContext::GetPrecisionMode() {
  auto op_precision_conf = OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  return op_precision_conf->precision_mode();
}

AscendDeviceProperties GeDeviceContext::GetDeviceProperties(uint32_t) {
  AscendDeviceProperties device_properties;
  const char *name = CALL_ASCEND_API(aclrtGetSocName);
  device_properties.name = (name == nullptr) ? "" : name;

  size_t free_size{0}, total_size{0};
  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &free_size, &total_size);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed get memory info for current device. Error number: " << ret;
  }
  device_properties.total_memory = total_size;
  device_properties.free_memory = free_size;
  return device_properties;
}

MS_REGISTER_DEVICE(kAscendDevice, GeDeviceContext);
#ifdef WITH_BACKEND
namespace {
void SetContextSocVersion(MsContext *ctx) {
  const std::map<std::string, std::string> kAscendSocVersions = {
    {"Ascend910A", "ascend910"},        {"Ascend910B", "ascend910"},        {"Ascend910PremiumA", "ascend910"},
    {"Ascend910ProA", "ascend910"},     {"Ascend910ProB", "ascend910"},     {"Ascend910B1", "ascend910b"},
    {"Ascend910B2", "ascend910b"},      {"Ascend910B2C", "ascend910b"},     {"Ascend910B3", "ascend910b"},
    {"Ascend910B4", "ascend910b"},      {"Ascend910_9391", "ascend910_93"}, {"Ascend910_9392", "ascend910_93"},
    {"Ascend910_9381", "ascend910_93"}, {"Ascend910_9382", "ascend910_93"}, {"Ascend910_9372", "ascend910_93"},
    {"Ascend910_9361", "ascend910_93"}, {"Ascend310P", "ascend310p"},       {"Ascend310P3", "ascend310p"},
    {"Ascend310B4", "ascend310b"},      {"Ascend310B1", "ascend310b"}};
  const char *soc_name_c = CALL_ASCEND_API(aclrtGetSocName);
  if (soc_name_c == nullptr) {
    MS_LOG(ERROR) << "Get soc name failed.";
    return;
  }
  std::string version(soc_name_c);
  MS_LOG(INFO) << "The soc version :" << version;
  ctx->set_ascend_soc_name(version);
  auto iter = kAscendSocVersions.find(version);
  if (iter == kAscendSocVersions.end()) {
    ctx->set_ascend_soc_version(version);
  } else {
    ctx->set_ascend_soc_version(iter->second);
  }
}
}  // namespace

MSCONTEXT_REGISTER_INIT_FUNC(kAscendDevice, [](MsContext *ctx) -> void {
  MS_EXCEPTION_IF_NULL(ctx);
  if (ctx->backend_policy() != "ge") {
    (void)ctx->set_backend_policy("ge");
  }
  // change some Environment Variables name
  auto format_mode = common::GetEnv("MS_ENABLE_FORMAT_MODE");
  if (!format_mode.empty()) {
    MS_LOG(WARNING)
      << "The Environment Variable MS_ENABLE_FORMAT_MODE will be discarded, please use MS_FORMAT_MODE instead.";
    common::SetEnv("MS_FORMAT_MODE", format_mode.c_str());
  }

  transform::LoadAscendApiSymbols();
  SetContextSocVersion(ctx);
});
#endif

// Register functions to _c_expression so python hal module could call Ascend device interfaces.
void PybindAscendStatelessFunc(py::module *m) {
  MS_EXCEPTION_IF_NULL(m);
  (void)py::class_<AscendDeviceProperties>(*m, "AscendDeviceProperties")
    .def_readonly("name", &AscendDeviceProperties::name)
    .def_readonly("total_memory", &AscendDeviceProperties::total_memory)
    .def_readonly("free_memory", &AscendDeviceProperties::free_memory)
    .def("__repr__", [](const AscendDeviceProperties &p) {
      std::ostringstream s;
      s << "AscendDeviceProperties(name='" << p.name << "', total_memory=" << p.total_memory / (1024 * 1024)
        << "MB, free_memory=" << p.free_memory / (1024 * 1024) << "MB)";
      return s.str();
    });
  (void)m->def("ascend_get_device_count", &GeDeviceContext::GetDeviceCount, "Get Ascend device count.");
  (void)m->def("ascend_get_device_name", &GeDeviceContext::GetDeviceName,
               "Get Ascend device name of specified device id.");
  (void)m->def("ascend_get_device_properties", &GeDeviceContext::GetDeviceProperties,
               "Get Ascend device properties of specified device id.");

  RegOpPrecisionConf(m);
  RegOpTuningConf(m);
  RegOpDebugConf(m);
}
REGISTER_DEV_STATELESS_FUNC_CB(kAscendDevice, PybindAscendStatelessFunc);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
