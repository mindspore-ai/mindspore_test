/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/actor_common.h"
#include <memory>
#include <unordered_map>
#include "mindspore/ops/op_def/framework_op_name.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_op_name.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/common/runtime_conf/runtime_conf.h"
#ifndef BUILD_LITE
#include "runtime/graph_scheduler/parameter_store.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/pipeline/runtime_pipeline.h"
#endif
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace runtime {
bool ActorDispatcher::is_multi_thread_execution_ = true;
bool ActorDispatcher::enable_multi_stream_ = false;
bool ActorDispatcher::has_kernel_need_user_data_ = false;
bool ActorDispatcher::is_memory_allocation_sync_ = true;
bool ActorDispatcher::is_memory_free_sync_ = true;
bool ActorDispatcher::enable_runtime_multi_pipeline_ = false;
bool ActorDispatcher::enable_async_launch_kernel_ = false;
bool ActorDispatcher::disable_kbk_sub_graph_execute_ = false;
bool ActorDispatcher::enable_sub_graph_execute_for_cur_actor_set_ = false;
bool ActorDispatcher::enable_static_shape_ = false;
bool ActorDispatcher::enable_trace_dynamic_memory_ = false;
bool ActorDispatcher::enable_use_trace_memory_ = false;
bool ActorDispatcher::enable_input_optimize_for_cur_actor_set_ = true;
bool ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set_ = false;
bool ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_step_ = false;

bool IsSuperKernelActor(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return (kernel_graph->is_graph_run_mode() &&
          ((node == nullptr) || node->isa<CNode>() || kernel_graph->IsChildGraphResult(node)));
}

bool IsRunningFailed(const OpContext<KernelTensor> *context) {
  if (UCEException::IsEnableUCE() || UCEException::GetInstance().enable_arf()) {
    if (UCEException::GetInstance().get_force_stop_flag() && !UCEException::GetInstance().get_has_throw_error()) {
      if (context->error_info_.empty()) {
        const_cast<OpContext<KernelTensor> *>(context)->error_info_ =
          std::string(UCEException::GetInstance().GetForceStopErrorMsg());
        MS_LOG(EXCEPTION) << UCEException::GetInstance().GetForceStopErrorMsg();
      }
    }
    if (UCEException::GetInstance().get_uce_flag() && !UCEException::GetInstance().get_has_throw_error()) {
      if (context->error_info_.empty()) {
        const_cast<OpContext<KernelTensor> *>(context)->error_info_ =
          std::string(UCEException::GetInstance().GetUceErrorMsg());
        MS_LOG(EXCEPTION) << UCEException::GetInstance().GetUceErrorMsg();
      }
    }
  }

  return (context->error_info_ != "");
}

bool IsHostQueueDSActor(const AnfNodePtr &node, const KernelGraphPtr &graph,
                        const std::vector<AnfNodePtr> &host_parameters, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);

  bool is_parameter_data = node->isa<Parameter>() && (!common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()));
  if (!is_parameter_data) {
    return false;
  }
  // Need to be updated every step.
  if (node->has_user_data(kForwardOutput)) {
    return true;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    MS_EXCEPTION_IF_NULL(graph);
    return graph->execution_order().size() > 1;
  }

  if (graph == nullptr) {
    return true;
  }

  // In control flow, only the parameters of the root funcgraph are in the host data source.
  const auto &front_node = graph->GetFrontAnfByBackendAnf(node);
  bool is_host = ((front_node == nullptr) ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  // Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
}

bool IsGraphRootParameter(const AnfNodePtr &node, const KernelGraphPtr &graph,
                          const std::vector<AnfNodePtr> &host_parameters, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(node);

  if (!node->isa<Parameter>()) {
    return false;
  }
  // Need to be updated every step.
  if (node->has_user_data(kForwardOutput)) {
    return true;
  }

  if (strategy == GraphExecutionStrategy::kStep) {
    MS_EXCEPTION_IF_NULL(graph);
    return graph->execution_order().size() > 1;
  }

  if (graph == nullptr) {
    return true;
  }

  // In control flow, only the parameters of the root funcgraph are in the host data source.
  const auto &front_node = graph->GetFrontAnfByBackendAnf(node);
  bool is_host = ((front_node == nullptr) ||
                  find(host_parameters.begin(), host_parameters.end(), front_node) != host_parameters.end());

  // Judge whether node is internal parameter.
  const auto &internal_front_node = graph->GetFrontNodeByInternalParameter(node);
  if (internal_front_node.first == nullptr && is_host) {
    return true;
  }

  return false;
}

bool IsSwitchActor(const AnfNodePtr &node) { return common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch); }

bool IsInternalParameter(const AnfNodePtr &node, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(graph);
  if (node->isa<Parameter>() && (!common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>()))) {
    //  Judge whether node is internal parameter.
    const auto &front_node = graph->GetOriginFrontNodeByInternalParameter(node);
    if (front_node.first != nullptr) {
      return true;
    }
  }
  return false;
}

bool IsKernelActor(const AnfNodePtr &node, GraphExecutionStrategy) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealCNodeKernel(node)) {
    return false;
  }

  return true;
}

bool IsSkippedKernelActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && common::AnfAlgo::IsInplaceNode(node, "skip")) {
    return true;
  }
  return false;
}

bool IsRpcActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && (common::AnfAlgo::GetCNodeName(node) == kRpcSendOpName ||
                              common::AnfAlgo::GetCNodeName(node) == kRpcRecvOpName)) {
    return true;
  }
  return false;
}

bool IsInnerControlFlowActor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsKernelActor(node) && (common::AnfAlgo::GetCNodeName(node) == "ConditionSwitch" ||
                              common::AnfAlgo::GetCNodeName(node) == "ConditionGather")) {
    return true;
  }
  return false;
}

bool IsPersistentDeviceTensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    return true;
  }

  // Maybe the load node, need fetch the real parameter node.
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (real_node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(real_node->cast<ParameterPtr>())) {
    return true;
  }
  return false;
}

bool IsControlFlowActor(KernelTransformType actor_type) {
  return ((actor_type >= KernelTransformType::kSwitchActor) && (actor_type <= KernelTransformType::kStackActor));
}

bool IsMemoryActor(KernelTransformType actor_type) {
  return ((actor_type == KernelTransformType::kMemoryAllocActor) ||
          (actor_type == KernelTransformType::kMemoryFreeActor));
}

bool IsSkippedLaunch(const CNodePtr &kernel, const KernelGraphPtr &kernel_graph) {
  static const char kLaunchSkippedEnv[] = "MS_KERNEL_LAUNCH_SKIP";
  static std::string launch_skipped = common::GetEnv(kLaunchSkippedEnv);
  static bool no_launch_skipped = launch_skipped.empty();
  if (no_launch_skipped) {
    return false;
  }

  static bool launch_skipped_all = (launch_skipped == "all" || launch_skipped == "ALL");
  if (launch_skipped_all) {
    MS_LOG(DEBUG) << "Skip all the launch.";
    return true;
  }

  std::string launch_name = "";
  std::string full_name = "";
  if (kernel != nullptr) {
    launch_name = common::AnfAlgo::GetCNodeName(kernel);
    full_name = kernel->fullname_with_scope();
  } else if (kernel_graph != nullptr) {
    launch_name = kernel_graph->ToString();
    full_name = kernel_graph->ToString();
  } else {
    return false;
  }

  if (launch_skipped == launch_name) {
    MS_LOG(DEBUG) << "Skip the launch of " << full_name;
    return true;
  }

  return false;
}

bool EnableAsyncInfer() {
  static const char kEnableAsyncInferdEnv[] = "MS_ENABLE_ASYNC_INFER";
  static bool ret = common::GetEnv(kEnableAsyncInferdEnv) == "1";
  return ret;
}

bool EnableTraceMemory() {
  static const bool enable_mem_tracker = common::IsEnableAllocConfig(common::kAllocMemoryTracker) ||
                                         !common::GetAllocConfigValue(common::kAllocMemoryTrackerPath).empty();
  if (enable_mem_tracker) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (!enable_infer_boost) {
    return false;
  }

  if (!EnableKbkSubGraphExecute()) {
    return false;
  }

  static const char kEnableTraceMemoryEnv[] = "MS_ENABLE_TRACE_MEMORY";
  static bool disable_trace_memory = common::GetEnv(kEnableTraceMemoryEnv) == "off";
  if (disable_trace_memory) {
    return false;
  }

  MS_LOG(INFO) << "Enable trace memory to optimize dynamic memory manage performance.";
  return true;
}

void ResetTraceMemoryStatus() {
  ActorDispatcher::set_enable_static_shape(false);
  ActorDispatcher::set_enable_trace_dynamic_memory(false);
  ActorDispatcher::set_enable_use_trace_memory(false);

  ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_actor_set(false);
  ActorDispatcher::set_enable_parallel_dispatch_kernel_for_cur_step(false);
}

void ResetPipelineStatus() {
  ActorDispatcher::set_enable_async_launch_kernel(false);
  ActorDispatcher::set_enable_runtime_multi_pipeline(false);
}

void ResetPipelineAndTraceMemoryStatus() {
  ResetPipelineStatus();
  ResetTraceMemoryStatus();
}

bool EnableKbkSubGraphExecute() {
  static bool disable_sub_graph_mode = common::IsDisableRuntimeConfig(common::kRuntimeKbkSubGraphMode);
  if (disable_sub_graph_mode) {
    return false;
  }

  if (ActorDispatcher::disable_kbk_sub_graph_execute()) {
    return false;
  }

  if (!EnableRuntimePipeline()) {
    return false;
  }

  if (!ActorDispatcher::enable_sub_graph_execute_for_cur_actor_set()) {
    return false;
  }

  return true;
}

bool EnableInputOptimize() {
  static bool disable_input_optimize = common::IsDisableRuntimeConfig(common::kRuntimeInputOptimize);
  if (disable_input_optimize) {
    return false;
  }

  if (!EnableKbkSubGraphExecute()) {
    return false;
  }

  if (!ActorDispatcher::enable_input_optimize_for_cur_actor_set()) {
    return false;
  }

  if (UCEException::IsEnableUCE() || UCEException::GetInstance().enable_arf()) {
    return false;
  }

  return true;
}

bool EnableRuntimePipeline() {
  static bool disable_runtime_pipeline = common::IsDisableRuntimeConfig(common::kRuntimePipeline);
  if (disable_runtime_pipeline) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return false;
  }

  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    return false;
  }

#ifndef BUILD_LITE
  if (distributed::recovery::RecoveryContext::GetInstance()->enable_recovery()) {
    return false;
  }
#endif

  return true;
}

bool EnableParallelDispatchKernel() {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  static bool enable_parallel_dispatch_kernel = runtime_conf_instance->IsKernelLaunchGroupConfigured();
  return enable_parallel_dispatch_kernel;
}

size_t GetDefragMemoryStepFreq() {
  static size_t defrag_memory_step_freq = 100L;

  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    MS_LOG(INFO) << "Init defrag memory step freq.";
    const auto &value = common::GetConfigValue(common::kAllocConf, common::kAllocDefragMemoryStepFreq);
    MS_LOG(INFO) << "Config defrag memory step freq : " << value << ".";
    if (value.size() != 0) {
      std::stringstream sstream(value);
      size_t config_value;
      sstream >> config_value;
      if (config_value != 0) {
        defrag_memory_step_freq = config_value;
      }
    }
    MS_LOG(INFO) << "Defrag memory step freq : " << defrag_memory_step_freq << ".";
  });

  return defrag_memory_step_freq;
}

bool WaitRuntimePipelineFinish(const OpContext<KernelTensor> *context, const std::string &name,
                               bool wait_kernel_launch_finish) {
#ifndef BUILD_LITE
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  if (ActorDispatcher::enable_runtime_multi_pipeline()) {
    if (EnableRuntimeNewPipeline()) {
      RuntimePipeline::GetInstance().infer_queue()->Wait();
      RuntimePipeline::GetInstance().resize_queue()->Wait();
    } else {
      KernelAsyncInferActor::GetInstance()->Wait();
      KernelAsyncResizeActor::GetInstance()->Wait();
    }
  }

  if (ActorDispatcher::enable_async_launch_kernel() && wait_kernel_launch_finish) {
    if (EnableRuntimeNewPipeline()) {
      RuntimePipeline::GetInstance().launch_queue()->Wait();
    } else {
      KernelAsyncLaunchActor::GetInstance()->Wait();
    }
  }
  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kWaitTaskFinish, name, false);

  if (ActorDispatcher::enable_async_launch_kernel() && IsRunningFailed(context)) {
    MS_LOG(INFO) << "Wait runtime pipeline finish and an error occurred: " << context->error_info_;
    return false;
  }
  return true;
#else
  return true;
#endif
}

bool Copy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor) {
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
  }

  // Exist the size alignment in some device, so get the min device size.
  size_t copy_size = std::min(src_device_tensor->GetSize(), dst_device_tensor->GetSize());
  auto skip_h2d = UCEException::GetInstance().is_reboot_node();

  if (dst_device_tensor->GetDeviceType() == src_device_tensor->GetDeviceType()) {
    return dst_device_tensor->SyncDeviceToDevice(src_device_tensor);
  } else if ((src_device_tensor->GetDeviceType() == device::DeviceType::kCPU) && !skip_h2d) {
    // CPU device tensor copy to other device tensor.
    return dst_device_tensor->SyncHostToDevice(copy_size, src_device_tensor->GetPtr());
  } else if (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // Other device tensor copy to CPU device tensor.
    return src_device_tensor->SyncDeviceToHost(copy_size, dst_device_tensor->GetMutablePtr());
  } else {
    MS_LOG(ERROR) << "Invalid device type, src device type: " << src_device_tensor->GetDeviceType()
                  << ", dst device type: " << dst_device_tensor->GetDeviceType();
    return false;
  }
}

bool AsyncCopy(const DeviceTensor *dst_device_tensor, const DeviceTensor *src_device_tensor, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  static const std::string kSyncCopyInput = "sync_copy_input";
  static bool sync_copy_input = common::IsEnableRuntimeConfig(kSyncCopyInput);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
  }

  // Exist the size alignment in some device, so get the min device size.
  size_t copy_size = std::min(src_device_tensor->GetSize(), dst_device_tensor->GetSize());

  MS_LOG(DEBUG) << "src device tensor type: " << src_device_tensor->GetDeviceType()
                << ", dst device tensor type: " << dst_device_tensor->GetDeviceType();
  bool ret = false;
  if (dst_device_tensor->GetDeviceType() == src_device_tensor->GetDeviceType()) {
    ret = dst_device_tensor->AsyncDeviceToDevice(src_device_tensor, stream_id);
  } else if (src_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // CPU device tensor copy to other device tensor.
    ret = dst_device_tensor->AsyncHostToDevice(copy_size, src_device_tensor->GetPtr(), stream_id);
  } else if (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // Other device tensor copy to CPU device tensor.
    // Use Sync instead of Async because cpu ops may use host ptr immediately.
    ret = src_device_tensor->SyncDeviceToHost(copy_size, dst_device_tensor->GetMutablePtr());
  } else {
    MS_LOG(ERROR) << "Invalid device type, src device type: " << src_device_tensor->GetDeviceType()
                  << ", dst device type: " << dst_device_tensor->GetDeviceType();
    return false;
  }
  if (sync_copy_input) {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {dst_device_tensor->device_name(), dst_device_tensor->device_id()});
    MS_EXCEPTION_IF_CHECK_FAIL(device_context->device_res_manager_->SyncAllStreams(), "Synchronize stream failed.");
  }
  return ret;
}

void FreeMemoryByDeviceContext(DeviceTensor *const device_tensor, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  // The device context may be not accurate in the control flow scene, so need fetch by device name and device id.
  if ((device_context == nullptr) || (device_context->GetDeviceType() != device_tensor->GetDeviceType())) {
    const auto &new_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(new_device_context);
    new_device_context->device_res_manager_->FreeMemory(device_tensor);
  } else {
    device_context->device_res_manager_->FreeMemory(device_tensor);
  }
}

void FreeMemoryByValueNode(const std::vector<std::weak_ptr<ValueNode>> &held_by_nodes, DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  device_tensor->ClearHeldByNodes();

  for (auto &node : held_by_nodes) {
    auto value_node = node.lock();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    tensor->set_device_address(nullptr);
    runtime::DeviceTensorStore::GetInstance().Remove(value_node.get());
  }
}

KernelTransformType FetchKernelTransformType(const AnfNodePtr &node, const KernelGraphPtr &graph,
                                             const std::vector<AnfNodePtr> &host_parameters,
                                             GraphExecutionStrategy strategy) {
  // Fetch kernel graph.
  KernelGraphPtr kernel_graph = nullptr;
  if (graph == nullptr) {
    kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return KernelTransformType::kUnknown;
  }
  if (kernel_graph->is_any_type_input() && node != nullptr && node->isa<CNode>()) {
    return KernelTransformType::kAnyTypeKernelActor;
  }
  // In sink mode, the data exchange between child graphs is expressed as parameters. These parameters are stored
  // in the graph and should be obtained from the super kernel actor.
  if (IsSuperKernelActor(node, kernel_graph)) {
    return KernelTransformType::kSuperKernelActor;
  }

  KernelTransformType type = KernelTransformType::kUnknown;
  MS_EXCEPTION_IF_NULL(node);
  auto real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_node);

  if (EnableInputOptimize()) {
    if (IsGraphRootParameter(real_node, kernel_graph, host_parameters, strategy)) {
      return KernelTransformType::kGraphParameterStore;
    }
  }

  if (IsHostQueueDSActor(real_node, kernel_graph, host_parameters, strategy)) {
    type = KernelTransformType::kHostDataSourceActor;
  } else if (IsKernelActor(real_node, strategy)) {
    type = KernelTransformType::kKernelActor;
  } else if (IsInternalParameter(real_node, kernel_graph)) {
    type = KernelTransformType::kInternalParameter;
  } else if (IsPersistentDeviceTensor(real_node)) {
    type = KernelTransformType::kDeviceTensorStore;
  } else {
    // May exist the from kernel that no need link in the pynative mode.
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Invalid from kernel: " << node->DebugString();
  }

  return type;
}

std::string FetchActorName(KernelTransformType kernel_type, const std::string &actor_set_name, const AnfNodePtr &node,
                           const KernelGraphPtr &graph) {
  // Fetch kernel graph.
  KernelGraphPtr kernel_graph = nullptr;
  if (graph == nullptr) {
    kernel_graph = AnfAlgo::FetchKernelGraph(node.get());
  } else {
    kernel_graph = graph;
  }
  if (kernel_graph == nullptr) {
    return "";
  }

  auto real_node = node;
  if (real_node != nullptr) {
    real_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({node, 0}).first;
  }
  std::string actor_name = "";
  switch (kernel_type) {
    case KernelTransformType::kSuperKernelActor:
      actor_name = kernel_graph->ToString() + kSuperKernelActorNameSuffix;
      break;
    case KernelTransformType::kAnyTypeKernelActor:
      actor_name = kernel_graph->ToString() + kAnyTypeKernelActorNameSuffix;
      break;
    case KernelTransformType::kHostDataSourceActor:
      actor_name = actor_set_name + kHostDSActorNameSuffix;
      break;
    case KernelTransformType::kGraphParameterStore:
      actor_name = actor_set_name + kReplaceDSActorStore;
      break;
    case KernelTransformType::kKernelActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = GetActorIdByKernel(real_node);
      break;
    case KernelTransformType::kKernelInferActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = kKernelInferActorNamePrefix + real_node->fullname_with_scope();
      break;
    case KernelTransformType::kKernelResizeActor:
      MS_EXCEPTION_IF_NULL(real_node);
      actor_name = kKernelResizeActorNamePrefix + real_node->fullname_with_scope();
      break;
    default:
      break;
  }
  return actor_name;
}

std::set<size_t> FetchModifiableRefInputIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Only the auto moand node will modify the input.
  if (!common::AnfAlgo::HasMonadInput(cnode)) {
    return {};
  }

  std::set<size_t> ref_input_indexes;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto &input = cnode->inputs().at(i);
    if (common::AnfAlgo::HasAbstractRef(input)) {
      (void)ref_input_indexes.insert(i - 1);
    }
  }

  return ref_input_indexes;
}

std::set<size_t> FetchModifiableRefOutputIndex(const CNodePtr &cnode, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(graph);
  std::set<size_t> ref_output_indexes;

  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    session::AnfWithOutIndex output_pair(cnode, i);
    // Only the ref node will modify the ref input corresponding to the output.
    if (!graph->IsInRefOutputMap(output_pair)) {
      continue;
    }
    auto input_pair = graph->GetRefCorrespondOutput(output_pair);
    MS_EXCEPTION_IF_NULL(input_pair.first);
    if (common::AnfAlgo::HasAbstractRef(input_pair.first)) {
      (void)ref_output_indexes.insert(i);
    }
  }
  return ref_output_indexes;
}

bool is_embedding_cache_server() {
  return ps::PSContext::instance()->cache_enable() && ps::PSContext::instance()->is_server();
}

void MemoryTraceManager::ReserveKernelMemoryBlocks(size_t size, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  (*kernel_memory_trace_blocks_)[device_context].reserve(size);
}

void MemoryTraceManager::PickMemoryTrackInfoForGraph(uint32_t graph_id) {
  if (graph_to_kernel_memory_trace_blocks_.find(graph_id) == graph_to_kernel_memory_trace_blocks_.end()) {
    graph_to_kernel_memory_trace_blocks_.emplace(
      graph_id, std::make_shared<std::map<const DeviceContext *, std::vector<KernelMemoryTraceBlockPtr>>>());
  }
  kernel_memory_trace_blocks_ = graph_to_kernel_memory_trace_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_memory_trace_blocks_);

  if (graph_to_merged_memory_trace_blocks_.find(graph_id) == graph_to_merged_memory_trace_blocks_.end()) {
    graph_to_merged_memory_trace_blocks_.emplace(
      graph_id, std::make_shared<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>>());
  }
  merged_memory_trace_blocks_ = graph_to_merged_memory_trace_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(merged_memory_trace_blocks_);

  if (graph_to_kernel_blocks_.find(graph_id) == graph_to_kernel_blocks_.end()) {
    graph_to_kernel_blocks_.emplace(
      graph_id, std::make_shared<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>>());
  }
  kernel_to_block_ = graph_to_kernel_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_to_block_);

  if (graph_to_kernel_tensor_with_mem_blocks_.find(graph_id) == graph_to_kernel_tensor_with_mem_blocks_.end()) {
    graph_to_kernel_tensor_with_mem_blocks_.emplace(
      graph_id, std::make_shared<HashMap<kernel::KernelTensor *, KernelMemoryTraceBlockPtr>>());
  }
  kernel_tensor_to_kernel_mem_blocks_ = graph_to_kernel_tensor_with_mem_blocks_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_tensor_to_kernel_mem_blocks_);
}

void MemoryTraceManager::AddKernelMemoryTraceBlock(const KernelMemoryTraceBlockPtr &block,
                                                   const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(block);
  MS_EXCEPTION_IF_NULL(block->start_);
  MS_EXCEPTION_IF_NULL(block->end_);
  (*kernel_memory_trace_blocks_)[device_context].emplace_back(block);
}

const std::shared_ptr<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>>
  &MemoryTraceManager::GetMergeBlocks() {
  return merged_memory_trace_blocks_;
}

const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>>
  &MemoryTraceManager::GetAllKernelBlocksnfo() {
  return kernel_to_block_;
}

const std::shared_ptr<HashMap<kernel::KernelTensor *, KernelMemoryTraceBlockPtr>>
  &MemoryTraceManager::GetKernelTensorToMemBlocksInfo() const {
  return kernel_tensor_to_kernel_mem_blocks_;
}

void MemoryTraceManager::MergeBlocks() {
  merged_memory_trace_blocks_->clear();
  for (auto &item : *kernel_memory_trace_blocks_) {
    auto &device_context = item.first;
    auto &kernel_memory_trace_blocks = item.second;
    MergeBlocksForSameDeviceContext(&kernel_memory_trace_blocks, &((*merged_memory_trace_blocks_)[device_context]));
    MS_LOG(DEBUG) << "The number of merged blocks is " << (*merged_memory_trace_blocks_)[device_context].size()
                  << ", device type: " << device_context->device_context_key().device_name_;
  }
}

void MemoryTraceManager::MergeBlocksForSameDeviceContext(
  std::vector<KernelMemoryTraceBlockPtr> *kernel_memory_trace_blocks,
  std::vector<MemoryTraceBlockPtr> *merged_memory_trace_blocks) {
  MS_EXCEPTION_IF_NULL(kernel_memory_trace_blocks);
  MS_EXCEPTION_IF_NULL(merged_memory_trace_blocks);
  merged_memory_trace_blocks->clear();

  if (kernel_memory_trace_blocks->empty()) {
    MS_LOG(INFO) << "No block to merge.";
    return;
  }

  std::sort(kernel_memory_trace_blocks->begin(), kernel_memory_trace_blocks->end(),
            [](const KernelMemoryTraceBlockPtr &block1, const KernelMemoryTraceBlockPtr &block2) {
              return (block1->start_ < block2->start_) ||
                     ((block1->start_ == block2->start_) && (block1->end_ < block2->end_));
            });
  merged_memory_trace_blocks->emplace_back(std::make_shared<MemoryTraceBlock>((*kernel_memory_trace_blocks)[0]->start_,
                                                                              (*kernel_memory_trace_blocks)[0]->size_));
  (*kernel_memory_trace_blocks)[0]->in_memory_trace_block_index_ = 0;
  for (size_t i = 1; i < kernel_memory_trace_blocks->size(); i++) {
    auto &back = merged_memory_trace_blocks->back();
    auto &block = (*kernel_memory_trace_blocks)[i];
    if (block->start_ >= back->end_) {
      merged_memory_trace_blocks->emplace_back(std::make_shared<MemoryTraceBlock>(block->start_, block->size_));
    } else if (block->end_ > back->end_) {
      back->end_ = block->end_;
      back->size_ = back->end_ - back->start_;
    }
    block->in_memory_trace_block_index_ = merged_memory_trace_blocks->size() - 1;
  }

  // Reset offset
  for (size_t i = 0; i < kernel_memory_trace_blocks->size(); i++) {
    auto &kernel_mem_block = (*kernel_memory_trace_blocks)[i];
    MS_EXCEPTION_IF_NULL(kernel_mem_block);
    const auto &mem_block = (*merged_memory_trace_blocks)[kernel_mem_block->in_memory_trace_block_index_];
    MS_EXCEPTION_IF_NULL(mem_block);
    if (kernel_mem_block->start_ < mem_block->start_) {
      MS_LOG(EXCEPTION) << "Invalid memory block, block start: " << kernel_mem_block->start_
                        << ", block end: " << kernel_mem_block->end_ << ", mem block start: " << mem_block->start_
                        << ", mem block end: " << mem_block->end_;
    }

    kernel_mem_block->offset_in_memory_trace_block_ = kernel_mem_block->start_ - mem_block->start_;
    (*kernel_to_block_)[kernel_mem_block->kernel_].emplace_back(kernel_mem_block);
    if (EnableParallelDispatchKernel() && kernel_mem_block->mem_type_ == kOutputMem) {
      kernel_tensor_to_kernel_mem_blocks_->emplace(kernel_mem_block->kernel_tensor_, kernel_mem_block);
    }
  }
}

void MemoryTraceManager::ClearExpiredCache() {
  kernel_memory_trace_blocks_->clear();
  merged_memory_trace_blocks_->clear();
  kernel_to_block_->clear();
  if (EnableParallelDispatchKernel()) {
    kernel_tensor_to_kernel_mem_blocks_->clear();
  }
}

void MemoryTraceManager::ClearAllCache() {
  for (auto &item : graph_to_kernel_memory_trace_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_memory_trace_blocks_.clear();

  for (auto &item : graph_to_merged_memory_trace_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_merged_memory_trace_blocks_.clear();

  for (auto &item : graph_to_kernel_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_blocks_.clear();

  for (auto &item : graph_to_kernel_tensor_with_mem_blocks_) {
    if (item.second) {
      item.second->clear();
    }
  }
  graph_to_kernel_tensor_with_mem_blocks_.clear();

  kernel_memory_trace_blocks_ = nullptr;
  merged_memory_trace_blocks_ = nullptr;
  kernel_to_block_ = nullptr;
  kernel_tensor_to_kernel_mem_blocks_ = nullptr;
}

std::unordered_map<AnfNode *, std::string> actor_ids;
static size_t actor_index = 0;

std::string GetActorIdByKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (actor_ids.find(node.get()) == actor_ids.end()) {
    MS_LOG(INFO) << "Cannot get actor id by node:" << node->fullname_with_scope();
    return node->fullname_with_scope();
  }
  return actor_ids[node.get()];
}

std::string GenerateActorIdByKernel(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto id = std::to_string(actor_index++) + "_" + node->fullname_with_scope();
  actor_ids[node.get()] = id;
  return id;
}

mindspore::HashMap<size_t, size_t> GetRepeatDeviceAddressIndexPair(const std::vector<KernelTensorPtr> &kernel_tensors) {
  mindspore::HashMap<const void *, std::vector<size_t>> ptr_positions;
  mindspore::HashMap<size_t, size_t> repeat_index;
  for (size_t i = 0; i < kernel_tensors.size(); ++i) {
    if (kernel_tensors[i] != nullptr && kernel_tensors[i]->device_address() != nullptr &&
        kernel_tensors[i]->device_address()->GetPtr() != nullptr) {
      ptr_positions[kernel_tensors[i]->device_address()->GetPtr()].emplace_back(i);
    }
  }
  for (const auto &pair : ptr_positions) {
    if (pair.second.size() <= 1) {
      continue;
    }
    for (size_t i = 1; i < pair.second.size(); ++i) {
      repeat_index[pair.second[i]] = pair.second[0];
    }
  }
  return repeat_index;
}

bool IsInferPhase(const std::string &phase) {
  return phase.find("prefill") != std::string::npos || phase.find("increment") != std::string::npos;
}

size_t FetchInputTensorIndex(const KernelWithIndex &front_node) {
  MS_EXCEPTION_IF_NULL(front_node.first);
  if (common::AnfAlgo::IsDynamicSequence(front_node.first)) {
    return 0;
  }

  const auto &abs = front_node.first->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    return front_node.second;
  }

  return 0;
}

TensorPtr FetchInputTensorByArg(const VectorRef &args, size_t arg_index, const KernelWithIndex &front_node) {
  if (arg_index >= args.size()) {
    MS_LOG(INFO) << "Arg index out of args range, index is " << arg_index << " and args size is " << args.size();
    return nullptr;
  }

  std::vector<tensor::TensorPtr> flatten_tensors;
  AnfAlgo::FlattenInputArg(args[arg_index], front_node.first, &flatten_tensors);
  auto input_tensor_index = FetchInputTensorIndex(front_node);
  if (input_tensor_index >= flatten_tensors.size()) {
    MS_LOG(INFO) << "Input tensor index out of args range, index is " << input_tensor_index << " and tensors size is "
                 << flatten_tensors.size();
    return nullptr;
  }

  return flatten_tensors[input_tensor_index];
}

bool IsEmptySequenceTensor(tensor::Tensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    return false;
  }
  const auto &sequence_shape = tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  return sequence_shape->size() == 0;
}

void UpdateDynamicShapeAndSize(tensor::Tensor *input_tensor, const KernelTensorPtr &kernel_tensor, size_t outer_index,
                               size_t inner_index) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (input_tensor == nullptr || IsEmptySequenceTensor(input_tensor)) {
    return;
  }

  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  if (!IsDynamic(kernel_tensor->host_shape()) && !graph_parameter_store->IsPositionDynamic(outer_index, inner_index)) {
    MS_LOG(DEBUG) << "No need to update dynamic shape and size, host shape dynamic is "
                  << IsDynamic(kernel_tensor->host_shape()) << ", graph parameter store outer index: " << outer_index
                  << ", inner index: " << inner_index << ", dynamic is "
                  << graph_parameter_store->IsPositionDynamic(outer_index, inner_index);
    return;
  }

  // Update shape.
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    MS_LOG(DEBUG) << "Kernel tensor: " << kernel_tensor.get() << ", shape is " << kernel_tensor->GetShapeVector();
    return;
  }
  kernel_tensor->SetShape(input_tensor->base_shape_ptr());
  MS_LOG(DEBUG) << "Kernel tensor: " << kernel_tensor.get() << ", shape is " << kernel_tensor->GetShapeVector();

  // Update size.
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto device_format = device_tensor->format();
  static const std::set<std::string> kNormalFormat = {
    kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->data().nbytes();
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Set device address:" << device_tensor << " size from:" << device_tensor->GetSize()
      << " to:" << tensor_data_size;
    device_tensor->SetSize(tensor_data_size);
  } else {
    MS_LOG(EXCEPTION) << "Can not Update size for 5D format device address";
  }
}

bool CopyDataFromTensor(const DeviceTensorPtr &device_tensor, tensor::Tensor *tensor, size_t stream_id) {
  static const std::string kSyncCopyInput = "sync_copy_input";
  static bool sync_copy_input = common::IsEnableRuntimeConfig(kSyncCopyInput);
  auto tensor_size = LongToSize(tensor->data().nbytes());
  auto ret = device_tensor->AsyncHostToDevice(tensor_size, tensor->data_type(), tensor->data_ptr(),
                                              tensor->device_info().host_format_, stream_id);

  if (sync_copy_input) {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_EXCEPTION_IF_CHECK_FAIL(device_context->device_res_manager_->SyncAllStreams(), "Synchronize stream failed.");
  }
  return ret;
}

void SyncHostToDeviceFromTensor(size_t outer_index, size_t inner_index, tensor::Tensor *tensor, const AID &from_aid,
                                const AnfNodePtr &node, bool is_first_user, size_t stream_id) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kKernelPrepareData, from_aid.Name());
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  if (NeedRunMemTracker()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), node->fullname_with_scope(),
                                                   from_aid.Name(), false);
  }
  // Update dynamic shape and size.
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  const auto &device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (graph_parameter_store->GetUserCnt(outer_index, inner_index) == 0) {
    MS_LOG(DEBUG) << "Skip sync host to device for kernel tensor:" << kernel_tensor->ToString()
                  << " outer index:" << outer_index << " inner index:" << inner_index << " for user count:0.";
    return;
  }
  UpdateDynamicShapeAndSize(tensor, kernel_tensor, outer_index, inner_index);
  graph_parameter_store->ResetAddrRefCount(outer_index, inner_index);
  if (TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
    device_tensor->IncreaseNewRefCount(from_aid.Name());
    MS_LOG(DEBUG) << from_aid.Name() << " do not use input outer index: " << outer_index
                  << ", inner index: " << inner_index << ", address: " << device_tensor
                  << " from graph parameter store.";
    return;
  }
  if (device_tensor->GetSize() == 0) {
    // The device tensor will not allocate a valid ptr, but it would be send to actor to decrease the ref count,
    // so the ref count should be add.
    device_tensor->IncreaseNewRefCount(from_aid.Name());
    MS_LOG(DEBUG) << from_aid.Name() << " input size is 0, outer index" << outer_index
                  << ", inner index: " << inner_index << ", address: " << device_tensor << ".";
    return;
  }

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->device_name(), device_tensor->device_id()});

  if (device_tensor->GetPtr() == nullptr) {
    auto mem_type = device_tensor->new_ref_count() == SIZE_MAX ? memory::mem_pool::MemType::kWeight
                                                               : memory::mem_pool::MemType::kKernel;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), mem_type, device_tensor->GetSize(),
                                                   device_tensor.get());
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, outer index: " << outer_index << ", inner index: " << inner_index
                        << ", for kernel tensor: " << kernel_tensor->ToString();
    }
  }

  auto tensor_size = LongToSize(tensor->data().nbytes());
  if (is_first_user) {
    if (tensor_size > 0 && !CopyDataFromTensor(device_tensor, tensor, stream_id)) {
      MS_LOG(EXCEPTION) << "Fetch parameter async host to device failed.";
    }
  } else if (graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index) == nullptr) {
    graph_parameter_store->SetAsyncMemcpyFun(
      outer_index, inner_index, [tensor_size, device_tensor, tensor](size_t stream_id) {
        if (tensor_size > 0 && !CopyDataFromTensor(device_tensor, tensor, stream_id)) {
          MS_LOG(EXCEPTION) << "Fetch parameter async host to device failed.";
        }
      });
  }

  graph_parameter_store->InsertTensorDataIntoCallback(tensor->data_ptr());
}

void SyncDataForTensorAddress(tensor::Tensor *tensor, const AID &from_aid, const AnfNodePtr &node) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kKernelPrepareData, from_aid.Name());
  if (NeedRunMemTracker()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), node->fullname_with_scope(),
                                                   from_aid.Name(), false);
  }

  const auto &tensor_address = std::static_pointer_cast<DeviceTensor>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(tensor_address);
  if (TEST_FLAG(tensor_address->flag(), device::kDeviceAddressFlagNotUsed)) {
    MS_LOG(DEBUG) << from_aid.Name() << " do not use the input.";
    return;
  }
  if (tensor_address->GetSize() == 0) {
    // The device tensor will not allocate a valid ptr, but it would be send to actor to decrease the ref count,
    // so the ref count should be add.
    MS_LOG(DEBUG) << from_aid.Name() << " input size is 0.";
    return;
  }

  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {tensor_address->device_name(), tensor_address->device_id()});
  if (tensor_address->GetPtr() == nullptr) {
    auto mem_type = tensor_address->new_ref_count() == SIZE_MAX ? memory::mem_pool::MemType::kWeight
                                                                : memory::mem_pool::MemType::kKernel;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), mem_type, tensor_address->GetSize(),
                                                   tensor_address.get());
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    if (!device_context->device_res_manager_->AllocateMemory(tensor_address.get(), kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed for tensor address: " << tensor_address->ToString();
    }
  }

  auto tensor_size = LongToSize(tensor->data().nbytes());
  if (tensor_size > 0 &&
      !tensor_address->SyncHostToDevice(tensor_address->GetShapeVector(), tensor_size, tensor->data_type(),
                                        tensor->device_info().host_format_, tensor->data_ptr())) {
    MS_LOG(EXCEPTION) << "Sync host to device for tensor address failed for tensor address: "
                      << tensor_address->ToString();
  }
}

void PrepareForNonTensorAddress(const std::pair<KernelWithIndex, size_t> &parameter_index, Tensor *tensor,
                                const AID &from_aid, bool is_first_user, size_t stream_id) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  if (device_tensor == nullptr) {
    abstract::BaseShapePtr shape;
    if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
      shape = tensor->ToAbstract()->GetShape();
    } else {
      shape = tensor->base_shape_ptr();
    }
    MS_EXCEPTION_IF_NULL(shape);
    auto old_addr_info_ret = graph_parameter_store->GetReleasePositionInfo({outer_index, inner_index});
    if (!old_addr_info_ret.first) {
      MS_LOG(EXCEPTION) << "Can not find info, outer index: " << outer_index << ", inner index: " << inner_index;
    }
    auto old_addr_info = old_addr_info_ret.second;
    TypePtr type = old_addr_info.first;
    MS_EXCEPTION_IF_NULL(type);
    auto device_name = graph_parameter_store->GetParameterDeviceName(outer_index, inner_index);
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_name, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    auto new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress();
    auto new_kernel_tensor =
      std::make_shared<kernel::KernelTensor>(new_device_tensor, shape, type, nullptr, ShapeVector{});
    new_kernel_tensor->set_size(LongToSize(tensor->data().nbytes()));
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Refresh store device tensor, from: " << new_device_tensor.get() << ", to null,"
      << ", outer index: " << outer_index << ", inner index: " << inner_index
      << ", device type: " << device::GetDeviceNameByType(new_device_tensor->GetDeviceType());
    new_device_tensor->SetNodeIndex(old_addr_info.second.first, old_addr_info.second.second);
    new_device_tensor->set_from_persistent_mem(true);
    kernel_tensor->set_device_address(new_device_tensor);
    device_tensor = new_device_tensor;
  }

  auto front_node = parameter_index.first;
  MS_EXCEPTION_IF_NULL(front_node.first);
  SyncHostToDeviceFromTensor(outer_index, inner_index, tensor, from_aid, front_node.first, is_first_user, stream_id);
  if ((graph_parameter_store->GetPositionWeight(outer_index) || common::AnfAlgo::HasAbstractRef(front_node.first))) {
    tensor->set_device_address(device_tensor);
    device_tensor->set_new_ref_count(SIZE_MAX);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set new ref count to max for device address:" << device_tensor;
  }
  graph_parameter_store->SetDeviceTensorPrepared(outer_index, inner_index, true);
}

void SetNodeIndexForTensorAddress(const DeviceTensorPtr &device_tensor, const DeviceTensorPtr &tensor_address,
                                  size_t outer_index, size_t inner_index) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  if (device_tensor != nullptr) {
    const auto &node_with_index = device_tensor->GetNodeIndex();
    tensor_address->SetNodeIndex(node_with_index.first, node_with_index.second);
    tensor_address->set_flag(device_tensor->flag());
  } else {
    auto old_addr_info_ret = graph_parameter_store->GetReleasePositionInfo({outer_index, inner_index});
    if (old_addr_info_ret.first) {
      auto old_addr_info = old_addr_info_ret.second;
      tensor_address->SetNodeIndex(old_addr_info.second.first, old_addr_info.second.second);
    }
  }
}

void PrepareParameter(const std::pair<KernelWithIndex, size_t> &parameter_index, const AID &from_aid,
                      bool is_first_user, size_t stream_id, bool enable_parallel_dispatch) {
  // Check parameter prepared for concurrent
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
    if (is_first_user && enable_parallel_dispatch) {
      const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
      if (copy_func != nullptr) {
        copy_func(stream_id);
        graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
      }
    }
    return;
  }
  auto front_node = parameter_index.first;
  MS_LOG(DEBUG) << "Prepare parameter input, actor: " << from_aid.Name() << ", outer index: " << outer_index
                << ", inner index:" << inner_index << ", front node: " << front_node.first->DebugString();
  auto tensor = graph_parameter_store->FetchTensor(outer_index, front_node);
  MS_EXCEPTION_IF_NULL(tensor);
  // Prepare data if got tensor address.
  auto tensor_address = std::static_pointer_cast<DeviceTensor>(tensor->device_address());
  if (tensor_address != nullptr) {
    graph_parameter_store->SetDeviceTensorPrepared(outer_index, inner_index, true);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set new ref count to max for device address:" << tensor_address;
    tensor_address->set_new_ref_count(SIZE_MAX);
    if (tensor_address->GetPtr() == nullptr) {
      // Tensor address may not from runtime, sync data with tensor.
      if (enable_parallel_dispatch) {
        MS_LOG(EXCEPTION) << "Not support parallel dispatch for tensor address with no device ptr.";
      }
      kernel_tensor->set_device_address(tensor_address);
      UpdateDynamicShapeAndSize(tensor, kernel_tensor, outer_index, inner_index);
      SyncDataForTensorAddress(tensor, from_aid, front_node.first);
      return;
    }

    auto device_tensor = kernel_tensor->device_address();
    if (tensor_address == device_tensor) {
      return;
    }

    // Set tensor address to kernel tensor.
    MS_LOG(DEBUG) << "Set tensor address to kernel tensor, tensor address: " << tensor_address->ToString()
                  << ", old device address: " << ((device_tensor == nullptr) ? "nullptr" : device_tensor->ToString())
                  << ", outer index: " << outer_index << ", inner index: " << inner_index
                  << ", kernel tensor: " << kernel_tensor->ToString();
    SetNodeIndexForTensorAddress(device_tensor, tensor_address, outer_index, inner_index);
    kernel_tensor->set_device_address(tensor_address);
    UpdateDynamicShapeAndSize(tensor, kernel_tensor, outer_index, inner_index);
    return;
  }

  // Prepare data for kernel tensor not from tensor.
  PrepareForNonTensorAddress(parameter_index, tensor, from_aid, is_first_user, stream_id);
}

KernelTensorPtr FetchParameter(const std::pair<KernelWithIndex, size_t> &parameter_index, const AID &from_aid,
                               bool is_first_user, size_t stream_id, bool enable_parallel_dispatch) {
  auto front_node = parameter_index.first.first;
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto outer_index = parameter_index.second;
  auto inner_index = parameter_index.first.second;
  graph_parameter_store->CheckIndexValid(outer_index, inner_index);
  MS_LOG(DEBUG) << "Fetch parameter for actor: " << from_aid.Name() << ", front node: " << front_node->DebugString()
                << ", with index: " << parameter_index.first.second << ", addr index: " << parameter_index.second;

  // The parameter is not concurrently used, do not use lock.
  if (!graph_parameter_store->IsConcurrentlyUse(outer_index, inner_index)) {
    // Return device tensor from graph parameter store if data prepared.
    auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
      // parallel dispatch kernel can not support multi graph parallel execute, no need lock.
      if (is_first_user && enable_parallel_dispatch) {
        const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
        if (copy_func != nullptr) {
          copy_func(stream_id);
          graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
        }
      }
      return kernel_tensor;
    }

    PrepareParameter(parameter_index, from_aid, is_first_user, stream_id, enable_parallel_dispatch);
    auto is_weight = graph_parameter_store->GetPositionWeight(outer_index);
    if (!is_weight && kernel_tensor->device_address() != nullptr &&
        kernel_tensor->device_address()->new_ref_count() == SIZE_MAX) {
      graph_parameter_store->InsertNonWeightRefMaxInputs(outer_index, inner_index);
    }
    return kernel_tensor;
  }

  // Return device tensor from graph parameter store if data prepared.
  static std::shared_mutex mtx;
  std::shared_lock<std::shared_mutex> read_lock(mtx);
  auto kernel_tensor = graph_parameter_store->Fetch(outer_index, inner_index);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (graph_parameter_store->GetDeviceTensorPrepared(outer_index, inner_index)) {
    // parallel dispatch kernel can not support multi graph parallel execute, no need lock.
    if (is_first_user && enable_parallel_dispatch) {
      const auto &copy_func = graph_parameter_store->GetAsyncMemcpyFun(outer_index, inner_index);
      if (copy_func != nullptr) {
        copy_func(stream_id);
        graph_parameter_store->SetAsyncMemcpyFun(outer_index, inner_index, nullptr);
      }
    }
    return kernel_tensor;
  }

  read_lock.unlock();
  std::unique_lock<std::shared_mutex> write_lock(mtx);
  PrepareParameter(parameter_index, from_aid, is_first_user, stream_id, enable_parallel_dispatch);
  auto is_weight = graph_parameter_store->GetPositionWeight(outer_index);
  if (!is_weight && kernel_tensor->device_address() != nullptr &&
      kernel_tensor->device_address()->new_ref_count() == SIZE_MAX) {
    graph_parameter_store->InsertNonWeightRefMaxInputs(outer_index, inner_index);
  }
  return kernel_tensor;
}
}  // namespace runtime
}  // namespace mindspore
