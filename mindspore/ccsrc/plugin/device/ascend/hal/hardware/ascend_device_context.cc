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

#include "plugin/device/ascend/hal/hardware/ascend_device_context.h"
#include <tuple>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include "include/common/utils/utils.h"
#include "include/common/debug/common.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "plugin/res_manager/ascend/device_context_conf/op_debug_conf.h"
#include "plugin/res_manager/ascend/device_context_conf/op_precision_conf.h"
#include "plugin/res_manager/ascend/device_context_conf/op_tuning_conf.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_util.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "plugin/res_manager/cpu/cpu_mem_manager/cpu_memory_manager.h"
#include "debug/profiler/profiling.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/res_manager/ascend/hccl_adapter/hccl_adapter.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_compiler_symbol.h"
#include "kernel/ascend/availability/silent_check/ascend_silent_check.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "acl/acl_dump.h"
#include "debug/dump/tensordump_control.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr auto kOpDebugConfigFile = "ge_op_debug_config.ini";
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";

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

RunMode AscendDeviceContext::GetRunMode(const FuncGraphPtr &func_graph) const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (common::AnfAlgo::IsDynamicShapeFuncGraph(func_graph)) {
    if (AnfAlgo::GetBackend(func_graph) == kBackendGE) {
      MS_LOG(INFO) << "set dynamic shape RunMode::kGraphMode";
      return RunMode::kGraphMode;
    }
    MS_LOG(INFO) << "set dynamic shape RunMode::kKernelMode";
    auto set_ctx = [&context](bool task_sink, bool is_multi_graph_sink, bool enable_loop_sink) {
      context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
      context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, is_multi_graph_sink);
      context->set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, enable_loop_sink);
    };
    set_ctx(false, false, false);
    return RunMode::kKernelMode;
  }

  if (context->IsKByKExecutorMode()) {
    MS_LOG(INFO) << "RunMode::kKernelMode, graph: " << func_graph->ToString();
    return RunMode::kKernelMode;
  } else {
    MS_LOG(INFO) << "RunMode::kGraphMode, graph: " << func_graph->ToString();
    return RunMode::kGraphMode;
  }
}

void AscendDeviceContext::InitializeForAclop() const {
  if (initialized_aclop_) {
    return;
  }
  if (!UseSimulationApi()) {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{device::DeviceType::kAscend, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    res_manager->InitializeForGe();
  }
  // should be called after ge initialize.
  SetAclOpDebugOption();
  dump::TensorDumpStepManager::GetInstance().SetAclDumpCallbackReg(reinterpret_cast<void *>(acldumpRegCallback));
  initialized_aclop_ = true;
}

void AscendDeviceContext::Initialize() {
  GilReleaseWithCheck gil_release;
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    return;
  }

  MS_LOG(INFO) << "Start initializing device context.";
  if (UseSimulationApi()) {
    device::ascend::LoadSimulationApiSymbols();
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

  // set MS_CTX_ENABLE_GE_HETEROGENOUS true according to heterogeneous mode
  ms_context->set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);

  if (ms_context->GetBackend() == kBackendGE) {
    InitializeForAclop();
  }

  MS_EXCEPTION_IF_NULL(GetKernelExecutor(false));
  MS_EXCEPTION_IF_NULL(GetKernelExecutor(true));
  // DynamicKernelExecutor and KernenlExecutor should be equal for GE
  MS_EXCEPTION_IF_CHECK_FAIL(GetKernelExecutor(true) == GetKernelExecutor(false),
                             "GE dynamic KernelExecutor and KernenlExecutor is not Equal.");
  GetKernelExecutor(false)->Initialize();

  InitDump();
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

void AscendDeviceContext::Destroy() {
  if (!IsNeedDestroy()) {
    // The device context is copied from main process by fork
    MS_LOG(INFO) << "The device context is not initialized by current process, it doesn't need to be destroyed.";
    return;
  }

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

void AscendDeviceContext::InitDump() const {
  if (common::AnfAlgo::IsBackendGe()) {
    MS_LOG(INFO) << "In the ge backend, dump is initialized at the same time as the backend.";
    return;
  }
  auto &dump_parser = DumpJsonParser::GetInstance();
  dump_parser.Parse();
}

DeprecatedInterface *AscendDeviceContext::GetDeprecatedInterface() {
  // need lock when multi-threads
  if (deprecated_interface_ == nullptr) {
    deprecated_interface_ = std::make_unique<AscendDeprecatedInterface>();
  }
  return deprecated_interface_.get();
}

uint32_t AscendDeviceContext::GetDeviceCount() { return AscendHalManager::GetInstance().GetDeviceCount(); }

std::string AscendDeviceContext::GetDeviceName(uint32_t) {
  const char *name = CALL_ASCEND_API(aclrtGetSocName);
  std::string device_name = (name == nullptr) ? "" : name;
  return device_name;
}

uint32_t AscendDeviceContext::GetExecuteTimeout() {
  auto op_debug_conf = OpDebugConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_debug_conf);
  return op_debug_conf->execute_timeout();
}

std::string AscendDeviceContext::GetAoeJobType() {
  auto op_tuning_conf = OpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  return op_tuning_conf->aoe_job_type();
}

std::string AscendDeviceContext::GetPrecisionMode() {
  auto op_precision_conf = OpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  return op_precision_conf->precision_mode();
}

AscendDeviceProperties AscendDeviceContext::GetDeviceProperties(uint32_t) {
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

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
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
    {"Ascend310B4", "ascend310b"},      {"Ascend310B1", "ascend310b"},      {"Ascend310", "ascend310"}};
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

  device::ascend::LoadAscendApiSymbols();
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
  (void)m->def("ascend_get_device_count", &AscendDeviceContext::GetDeviceCount, "Get Ascend device count.");
  (void)m->def("ascend_get_device_name", &AscendDeviceContext::GetDeviceName,
               "Get Ascend device name of specified device id.");
  (void)m->def("ascend_get_device_properties", &AscendDeviceContext::GetDeviceProperties,
               "Get Ascend device properties of specified device id.");

  RegOpPrecisionConf(m);
  RegOpTuningConf(m);
  RegOpDebugConf(m);
}
REGISTER_DEV_STATELESS_FUNC_CB(kAscendDevice, PybindAscendStatelessFunc);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
