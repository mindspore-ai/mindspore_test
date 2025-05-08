/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/kernel_actor.h"

#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <limits>

#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/device/res_manager/multi_stream_controller.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "runtime/graph_scheduler/execution_order_check/kernel_cache.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/backend/debug/execute_order_tracker/execute_order_tracker.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "include/backend/distributed/collective/collective_manager.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "kernel/framework_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/compile_config.h"
#include "mindspore/ops/op_def/structure_op_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
namespace mindspore {
namespace runtime {
namespace {
bool IsSomasEnable(const SomasInfo *somas_info) {
  return ((somas_info != nullptr) && (somas_info->whole_block_size_ != 0));
}

void CheckDryRun(const CNodePtr &kernel_) {
  static const bool is_dry_run_mode =
    (common::IsCompileSimulation() || common::IsExecuteSimulation() || common::IsDryRun());
  static auto enabled_profile = common::GetCompileConfig("COMPILE_PROFILE") == "1";
  if (is_dry_run_mode && !enabled_profile) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_)
      << "The dry run mode can not support dynamic shape graph which contains value depend or computing depend kernel:"
      << kernel_->fullname_with_scope()
      << ", launch kernel is skipped for dry run mode, which leads to fail to GetValue for infer "
         "shape of these value depend or computing depend kernel. You can only simulate compile graph and not do "
         "InferShape and Resize by `export MS_SIMULATION_LEVEL=0` instead.";
  }
}
void TrackInputOutputMemory(const std::vector<KernelTensor *> &input_launch_tensors,
                            const std::vector<KernelTensor *> &output_launch_tensors, const std::string &actor_name,
                            const std::vector<bool> &depend_shape_input_list) {
  for (size_t i = 0, end = input_launch_tensors.size(); i < end; i++) {
    // Skip shape depend inputs.
    if (i < depend_shape_input_list.size() && depend_shape_input_list[i]) {
      continue;
    }
    auto device_addr = input_launch_tensors[i]->device_address().get();
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsInput, actor_name, device_addr->device_name(),
                                                   device_addr->GetPtr(), device_addr->type_id(),
                                                   device_addr->GetShapeVector(), device_addr->GetTensorStorageInfo());
  }
  for (size_t i = 0, end = output_launch_tensors.size(); i < end; i++) {
    auto device_addr = output_launch_tensors[i]->device_address().get();
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(MarkTensorAsOutput, actor_name, device_addr->device_name(),
                                                   device_addr->GetPtr(), device_addr->type_id(),
                                                   device_addr->GetShapeVector(), device_addr->GetTensorStorageInfo());
  }
}

void AddNodeMemTrackerInfo(const CNodePtr cnode, const std::string &actor_name, bool is_stream_recv_actor) {
  if (is_stream_recv_actor || IsPrimitiveCNode(cnode, prim::kPrimStreamSend)) {
    auto node_name = is_stream_recv_actor ? "WaitEvent" : "RecordEvent";
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, node_name, node_name, "", true);
  } else {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, actor_name, cnode->fullname_with_scope(),
                                                   cnode->func_graph()->ToString(), true);
  }
}

void AddNodeToGraphTracker(const CNodePtr cnode, const std::string &actor_name) {
  auto type = common::AnfAlgo::GetCNodeName(cnode);
  auto stream_id = std::to_string(AnfAlgo::GetStreamId(cnode));
  if (type == kStreamSendOpName || type == kStreamRecvOpName) {
    auto node_name = type == kStreamSendOpName ? "RecordEvent" : "WaitEvent";
    std::string event_id;
    if (common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode)) {
      event_id = std::to_string(common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrEventId));
    } else {
      MS_LOG(EXCEPTION) << "StreamSend or StreamRecv ops does not have attribute kAttrEventId.";
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, node_name, node_name, "", true);
    device::tracker::CALL_MEMORY_TRACKER(
      UpdateTask, node_name, {{device::tracker::kStreamId, stream_id}, {device::tracker::kEvent, event_id}});
  } else {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, actor_name, cnode->fullname_with_scope(),
                                                   cnode->func_graph()->ToString(), true);
    device::tracker::CALL_MEMORY_TRACKER(UpdateTask, actor_name, {{device::tracker::kStreamId, stream_id}});

    if (!(common::AnfAlgo::IsCommunicationOp(cnode) && common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode))) {
      return;
    }

    auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
    std::vector<uint32_t> comm_ranks;
    if (group_name == "hccl_world_group") {
      uint32_t rank_size = 1;
#if !defined(BUILD_LITE)
      rank_size = distributed::collective::CollectiveManager::instance()->global_rank_size();
#endif
      comm_ranks.resize(rank_size);
      std::iota(comm_ranks.begin(), comm_ranks.end(), 0);
    } else {
#if !defined(BUILD_LITE)
      comm_ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(group_name);
#else
      comm_ranks = {0};
#endif
    }
    std::string comm_ranks_str = std::accumulate(
      comm_ranks.begin(), comm_ranks.end(), std::string(),
      [](const std::string &a, uint32_t b) { return a.empty() ? std::to_string(b) : a + " " + std::to_string(b); });
    std::unordered_map<std::string, std::string> attrs = {{device::tracker::kGroup, group_name},
                                                          {device::tracker::kCommRank, comm_ranks_str}};

    auto get_rank = [&](const std::string &attr_name) -> uint32_t {
      uint32_t rank_value = std::numeric_limits<uint32_t>::max();
      if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
        int64_t rank_attr = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr_name);
        if (rank_attr >= 0 && static_cast<size_t>(rank_attr) < comm_ranks.size()) {
          rank_value = comm_ranks[static_cast<size_t>(rank_attr)];
        } else {
          MS_LOG(EXCEPTION) << "Invalid rank_attr value: " << rank_attr << ", or out of range for comm_ranks with size "
                            << comm_ranks.size() << ".";
        }
      }
      return rank_value;
    };
    auto src_rank = get_rank(device::tracker::kSrcRank);
    if (src_rank != std::numeric_limits<uint32_t>::max()) {
      attrs[device::tracker::kSrcRank] = std::to_string(src_rank);
    }
    auto dst_rank = get_rank(device::tracker::kDstRank);
    if (dst_rank != std::numeric_limits<uint32_t>::max()) {
      attrs[device::tracker::kDstRank] = std::to_string(dst_rank);
    }
    auto root_rank = get_rank(device::tracker::kRootRank);
    if (root_rank != std::numeric_limits<uint32_t>::max()) {
      attrs[device::tracker::kRootRank] = std::to_string(root_rank);
    }
    device::tracker::CALL_MEMORY_TRACKER(UpdateTask, actor_name, attrs);
  }
  return;
}
}  // namespace

using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

KernelActor::KernelActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                         const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                         GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                         const std::set<size_t> &modifiable_ref_output_indexes, const KernelTransformType &type)
    : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid, nullptr),
      kernel_(kernel),
      is_dynamic_value_(false),
      is_dynamic_type_(false),
      has_dynamic_(false),
      enable_async_infer_(false),
      kernel_info_(nullptr),
      kernel_mod_(nullptr),
      somas_info_(nullptr),
      real_input_num_(0),
      strategy_(strategy),
      modifiable_ref_input_indexes_(modifiable_ref_input_indexes),
      modifiable_ref_output_indexes_(modifiable_ref_output_indexes),
      is_launch_skipped_(false),
      inputs_continuous_memory_(false) {
  (void)device_contexts_.emplace_back(device_context);
  is_dynamic_shape_ = common::AnfAlgo::IsDynamicShape(kernel_) || common::AnfAlgo::IsDynamicSequence(kernel_);

  kernel_async_infer_aid_ = KernelAsyncInferActor::GetInstance()->GetAID();
  kernel_async_resize_aid_ = KernelAsyncResizeActor::GetInstance()->GetAID();
  kernel_async_launch_aid_ = KernelAsyncLaunchActor::GetInstance()->GetAID();
  input_free_index_.resize(common::AnfAlgo::GetInputTensorNum(kernel));
  std::iota(input_free_index_.begin(), input_free_index_.end(), 0);
  output_free_index_.resize(AnfAlgo::GetOutputAddressNum(kernel));
  std::vector<bool> is_output_kernel(AnfAlgo::GetOutputAddressNum(kernel), false);
  is_output_kernel_.swap(is_output_kernel);
  std::iota(output_free_index_.begin(), output_free_index_.end(), 0);
  MS_LOG(DEBUG) << "Input free index:" << input_free_index_ << " output free index:" << output_free_index_
                << " for actor:" << GetAID() << " kernel:" << kernel->DebugString();

  // shape depend need kernel is cnode.
  SetShapeDependInfo();
}

void KernelActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  MS_EXCEPTION_IF_NULL(kernel_);
  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // monad
  InitIsMonadInput();
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  is_dynamic_value_ = common::AnfAlgo::IsDynamicValue(kernel_);
  if (is_dynamic_shape_ && IsSomasEnable(somas_info_)) {
    MS_LOG(EXCEPTION) << "Not support the somas for the dynamic shape: " << GetAID().Name();
  }
  is_dynamic_type_ = common::AnfAlgo::IsAnyTypeOutput(kernel_);
  has_dynamic_ = is_dynamic_shape_ || is_dynamic_type_ || is_dynamic_value_;
  bool is_value_dyn = (is_dynamic_value_ && (is_dynamic_shape_ || is_dynamic_type_));
  if (is_value_dyn || (kernel_mod_->IsNeedUpdateOutputShapeAndSize() &&
                       no_dyn_need_update_ops.find(kernel_mod_->kernel_name()) == no_dyn_need_update_ops.end())) {
    CheckDryRun(kernel_);
  }

  // Check whether the kernel has input node which is a computed depend kernel.
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  auto kernel_executor = device_contexts_[0]->GetKernelExecutor(false);
  MS_EXCEPTION_IF_NULL(kernel_executor);
  launch_ignored_inputs_ = kernel_executor->GetLaunchIgnoredInputAddressIdx(kernel_);

  stream_ = device_contexts_[0]->device_res_manager_->GetStream(kernel_info_->stream_id());
  // Init the device tensors and kernel launch info.
  InitInputInfo();
  InitOutputInfo();
  InitWorkspaceInfo();

  // Set flag to check input contiguous
  if (kernel::NeedCheckInputContiguous(kernel_)) {
    need_check_tensor_contiguous_ = true;
  }

  // Init the output data.
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }
  size_t output_data_index = 0;
  for (auto &data_arrow : output_data_arrows_) {
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    data->data_ = output_kernel_tensors_[IntToSize(data_arrow->from_output_index_)];
    ++output_data_index;
  }
  this->InitMultiStreamInfo();
}

void KernelActor::InitMultiStreamInfo() {
  auto device_context = device_contexts_[0];
  // cpu kernel does not need multi stream process, and gpu kernel has not adapt it currently.
  if (device_context->GetDeviceType() == device::DeviceType::kCPU ||
      device_context->GetDeviceType() == device::DeviceType::kGPU) {
    MS_LOG(DEBUG) << "Kernel : " << kernel_->fullname_with_scope() << " device type is "
                  << device_context->GetDeviceType() << ", will skip multi stream process.";
    is_multi_stream_process_skipped_ = true;
  }

  // Share pointer of task id on stream with output kernel tensor.
  for (auto &output_kernel_tensor : output_kernel_tensors_) {
    output_kernel_tensor->set_task_id_on_stream(task_id_on_stream_);
  }
  is_stream_recv_actor_ = IsPrimitiveCNode(kernel_, prim::kPrimStreamRecv);
  // kernel_ may be ValueNode<FuncGraph>, skip exception situation.
  auto cnode = kernel_->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  constexpr char kRuntimeMc2Event[] = "mc2_event";
  bool match_mc2_pattern = std::string::npos != kernel_->fullname_with_scope().find("_all_gather_matmul") ||
                           std::string::npos != kernel_->fullname_with_scope().find("_matmul_reduce_scatter") ||
                           std::string::npos != kernel_->fullname_with_scope().find("MatmulReduceScatter-") ||
                           std::string::npos != kernel_->fullname_with_scope().find("AllGatherMatmul-") ||
                           std::string::npos != kernel_->fullname_with_scope().find("MatMulAllReduce-");
  is_mc2_kernel_ = (common::IsEnableRuntimeConfig(common::kRuntimeMultiStream)) &&
                   !common::IsDisableRuntimeConfig(kRuntimeMc2Event) && match_mc2_pattern;

  auto input0 = cnode->input(kAnfPrimitiveIndex);
  if (IsValueNode<FuncGraph>(input0)) {
    MS_LOG(INFO) << "Cnode is not a func graph value node : " << kernel_->fullname_with_scope() << ".";
    return;
  }

  auto multi_stream_safe_value = cnode->GetAttr(kAttrInputMultiStreamSafe);
  if (multi_stream_safe_value != nullptr) {
    is_multi_stream_safe_ = GetValue<bool>(multi_stream_safe_value);
    MS_LOG(DEBUG) << "cnode : " << cnode->DebugString() << " is thread safe.";
  }
}

void KernelActor::InitIsMonadInput() {
  auto build_info = kernel_info_->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  is_monad_input_.resize(real_input_num_, false);
  for (size_t i = 0; i < real_input_num_; ++i) {
    if (common::AnfAlgo::IsMonadType(build_info->GetInputDeviceType(i))) {
      is_monad_input_[i] = true;
    }
  }
}

void KernelActor::InitInputInfo() {
  for (size_t i = 0; i < real_input_num_; ++i) {
    if (is_monad_input_[i]) {
      auto build_info = kernel_info_->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      (void)real_input_data_infos_.emplace_back(std::make_shared<InputDataInfo>(
        build_info->GetInputFormat(i), ShapeVector{}, 0, build_info->GetInputDeviceType(i)));
      continue;
    }
    const auto &input_kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    const auto &input_device_tensor = input_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_device_tensor->format(), input_kernel_tensor->host_shape(),
                                      input_device_tensor->GetSize(), input_device_tensor->type_id()));
  }

  copy_input_kernel_tensors_.resize(real_input_num_);
  pre_input_kernel_tensors_.resize(real_input_num_);
  contiguous_tensors_.resize(real_input_num_);
  input_launch_tensors_.resize(real_input_num_);
  input_kernel_tensors_.resize(real_input_num_);
  input_kernel_tensors_for_infer_.resize(real_input_num_);
  for (auto &input_kernel_tensor : input_kernel_tensors_) {
    (void)memory_free_list_.emplace_back(input_kernel_tensor);
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.inputs_.emplace_back(std::make_shared<Address>());
    }
  }
  for (size_t index : input_free_index_) {
    if (index >= input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid input index:" << index << " in free index:" << input_free_index_
                        << " input size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_kernel_tensors_[index]);
  }
}

namespace {
void ResetNewRefCountForRefOutputInSomas(const CNodePtr &node, size_t index) {
  if (node == nullptr) {
    return;
  }
  auto kernel_info = dynamic_cast<KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return;
  }
  const auto &ref_map = kernel_info->out_in_ref_map();
  const auto &iter = ref_map.find(index);
  if (iter == ref_map.end()) {
    return;
  }
  size_t input_index = iter->second;
  if (index >= common::AnfAlgo::GetInputTensorNum(node)) {
    return;
  }
  const auto &input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_index, false);
  if (input_node_with_index.first == nullptr || !input_node_with_index.first->isa<CNode>() ||
      common::AnfAlgo::CheckPrimitiveType(input_node_with_index.first, prim::kPrimConditionGather) ||
      !AnfAlgo::OutputAddrExist(input_node_with_index.first, input_node_with_index.second, false)) {
    return;
  }
  const auto &input_device_tensor =
    AnfAlgo::GetMutableOutputAddr(input_node_with_index.first, input_node_with_index.second, false);
  input_device_tensor->set_new_ref_count(0);
  MS_LOG(DEBUG) << "Set new ref count to 0 for device tensor:" << input_device_tensor->PrintInfo()
                << " for node:" << input_node_with_index.first->fullname_with_scope()
                << " debug string:" << input_node_with_index.first->DebugString()
                << " index:" << input_node_with_index.second;
  ResetNewRefCountForRefOutputInSomas(input_node_with_index.first->cast<CNodePtr>(), input_node_with_index.second);
}
}  // namespace

void KernelActor::InitOutputInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  const auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  bool output_need_somas = false;
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto &output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto &output_address = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_address);

    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }

    (void)output_kernel_tensors_.emplace_back(output_kernel_tensor);
    (void)output_launch_tensors_.emplace_back(output_kernel_tensor.get());
    MS_LOG(DEBUG) << "Init output[" << i << "] info for node:" << kernel_->fullname_with_scope()
                  << " addr:" << output_address << " type:" << output_address->type_id()
                  << ", kernel tensor addr:" << output_kernel_tensor.get()
                  << ", kernel tensor: " << output_kernel_tensor->ToString();
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.outputs_.emplace_back(std::make_shared<Address>());
    }
    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      output_kernel_tensor->set_managed_by_somas(true);
      MS_LOG(INFO) << "Device address : " << output_address << ", kernel tensor : " << output_kernel_tensor
                   << " is managed by somas.";
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, output index:" << i
                     << " somas aligned size:" << somas_outputs[i].second
                     << " is smaller than address size:" << output_address->GetSize();
      }
      // Used to keep graph output address when somas block memory free, and reused by the ref conut in other graphs.
      if (somas_graph_output_indexes_.count(i) > 0) {
        MS_LOG(DEBUG) << "Somas keep output device address:" << output_address << " ptr:" << output_address->GetPtr();
        (void)somas_info_->InsertGraphOutputInfo(output_address.get(), somas_outputs[i].first, somas_outputs[i].second);
        ResetNewRefCountForRefOutputInSomas(kernel_, i);
      } else {
        UpdateRefCount(output_address.get(), true);
        output_address->set_new_ref_count(SIZE_MAX);
      }
      output_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(output_kernel_tensor);
      if (is_output_kernel_[i]) {
        max_ref_cnt_output_list_.emplace_back(output_kernel_tensor);
        MS_LOG(DEBUG) << "Add output kernel tensor:" << output_kernel_tensor << " for trace in actor:" << GetAID();
      }
      (void)memory_free_list_.emplace_back(output_kernel_tensor);
    }
  }

  for (size_t index : output_free_index_) {
    if (index >= output_kernel_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid output free index:" << index << " total size:" << output_kernel_tensors.size()
                        << " for actor:" << GetAID();
    }
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, index) || output_kernel_tensors[index] == nullptr) {
      continue;
    }
    MS_LOG(DEBUG) << "Add output free kernel tensor:" << output_kernel_tensors[index] << " for actor:" << GetAID();
    new_memory_free_list_.emplace_back(output_kernel_tensors[index]);
  }
  if (output_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((output_kernel_tensors_.size() >= somas_outputs.size()), "The output num is wrong.");
  }

  for (auto &external_reference_tensor : external_reference_tensors_) {
    (void)memory_free_list_.emplace_back(external_reference_tensor);
  }
}

void KernelActor::InitWorkspaceInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // The size of workspace maybe changed in dynamic shape, so put workspace_address in the end of memory_alloc_list_ and
  // memory_free_list_, for the operation of dynamic_shape condition in FetchWorkspaceDeviceTensor.
  const auto &workspace_kernel_tensor_list = kernel_info_->workspace_kernel_tensor_list();
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  bool workspace_need_somas = false;
  for (size_t i = 0; i < workspace_kernel_tensor_list.size(); ++i) {
    auto &workspace_kernel_tensor = workspace_kernel_tensor_list[i];
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensor);
    auto &workspace_address = workspace_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(workspace_address);
    (void)workspace_kernel_tensors_.emplace_back(workspace_kernel_tensor);
    (void)workspace_launch_tensors_.emplace_back(workspace_kernel_tensor.get());
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
    }

    // The workspace taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_workspace, i)) {
      if (somas_workspace[i].second < workspace_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, workspace index:" << i
                     << " somas aligned size:" << somas_workspace[i].second
                     << " is smaller than address size:" << workspace_address->GetSize();
      }
      UpdateRefCount(workspace_address.get(), true);
      workspace_address->set_new_ref_count(SIZE_MAX);
      workspace_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(workspace_kernel_tensor);
      (void)memory_free_list_.emplace_back(workspace_kernel_tensor);
      (void)new_memory_free_list_.emplace_back(workspace_kernel_tensor);
    }
  }

  if (workspace_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((workspace_kernel_tensors_.size() >= somas_workspace.size()),
                               "The output num is wrong.");
  }
}

void KernelActor::SetShapeDependInfo() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return;
  }
  // Shape kernel no need to decrease ref count.
  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(kernel_, kAttrOnlyDependShape);
  if (only_depend_shape_attr != nullptr) {
    auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
    MS_LOG(INFO) << "Init shape depend info, real_input_num_ : " << real_input_num_
                 << ", only_depend_shape size : " << only_depend_shape.size() << ".";
    for (size_t i = 0; i < only_depend_shape.size(); i++) {
      // shape depend, no need free this device tensor.
      MS_LOG(INFO) << "only_shape_depend[" << i << "] : " << only_depend_shape[i] << ".";
      depend_shape_input_list_.emplace_back(only_depend_shape[i]);
    }
  }
  if (depend_shape_input_list_.empty()) {
    return;
  }
  std::vector<size_t> need_free_input_index;
  for (size_t index : input_free_index_) {
    if (index < depend_shape_input_list_.size() && depend_shape_input_list_[index]) {
      MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip free input device tensor index:" << index;
      continue;
    }
    need_free_input_index.emplace_back(index);
  }
  input_free_index_.swap(need_free_input_index);
}

void KernelActor::ConvertInputContiguous(OpContext<KernelTensor> *const context) {
  auto cur_stream_id = device_contexts_[0]->device_res_manager_->GetCurrentStreamId();
  auto stream_id = kernel_info_->stream_id();
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto input_device_tensor = input_kernel_tensors_[i]->device_address().get();
    if (input_device_tensor == nullptr) {
      continue;
    }
    if (i >= contiguous_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), " input out of range.");
    }
    const auto old_storage_info = input_device_tensor->GetTensorStorageInfo();
    if (old_storage_info) {
      // input addr is contiguous and shape size is equal to origin.
      if ((SizeOf(old_storage_info->shape) == SizeOf(old_storage_info->ori_shape)) && old_storage_info->is_contiguous) {
        continue;
      }
      if (!launch_ignored_inputs_.empty() && (std::find(launch_ignored_inputs_.begin(), launch_ignored_inputs_.end(),
                                                        i) != launch_ignored_inputs_.end())) {
        MS_LOG(DEBUG) << GetAID().Name() << " ignore the input address for input index: " << i;
        continue;
      }
      MS_LOG(INFO) << "Make input [" << i << "] contiguous for kernel " << kernel_->DebugString();
      if (contiguous_tensors_[i] == nullptr) {
        // Make new device tensor and run InplaceCopy to make contiguous.
        MS_EXCEPTION_IF_NULL(old_storage_info);
        auto address_size = GetTypeByte(TypeIdToType(input_device_tensor->type_id())) * SizeOf(old_storage_info->shape);
        auto kernel_tensor = AnfAlgo::CreateKernelTensor(
          nullptr, address_size, Format::DEFAULT_FORMAT, input_device_tensor->type_id(), old_storage_info->shape,
          device_contexts_[0]->device_context_key().device_name_, device_contexts_[0]->device_context_key().device_id_);
        kernel_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(input_device_tensor->type_id())));
        kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(old_storage_info->shape));
        kernel_tensor->set_stream_id(stream_id);

        auto new_device_address = kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(new_device_address);
        new_device_address->set_device_shape(old_storage_info->shape);
        // Store the temp device address
        contiguous_tensors_[i] = kernel_tensor;
      }
      auto &new_kernel_tensor = contiguous_tensors_[i];
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      auto &new_device_address = new_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(new_device_address);
      if (is_dynamic_shape_) {
        auto input_tensor = input_kernel_tensors_[i];
        MS_EXCEPTION_IF_NULL(input_tensor);
        MS_EXCEPTION_IF_NULL(input_tensor->GetShape());
        new_kernel_tensor->SetShape(input_tensor->GetShape()->Clone());
        MS_EXCEPTION_IF_NULL(input_tensor->device_address());
        auto address_size =
          GetTypeByte(TypeIdToType(input_tensor->device_address()->type_id())) * SizeOf(old_storage_info->shape);
        new_kernel_tensor->set_size(address_size);
      }
      new_device_address->set_tensor_storage_info(nullptr);
      // Launch CopyInplace to make tensor contiguous.
      if (!device_contexts_[0]->GetKernelExecutor(false)->ExecuteKernelTask(
            runtime::KernelTaskType::kCONTIGUOUS_TASK, {input_device_tensor}, {new_device_address.get()}, stream_id)) {
        MS_LOG(EXCEPTION) << "Graph mode executeKernelTask Contiguous failed.";
      }
      // Store the old tensor storage info , input device tensor and input kernel tensor.
      // Recover them when launch finished.
      if (cur_stream_id != stream_id) {
        cross_stream_addresses_.emplace_back(0, input_kernel_tensors_[i]->device_ptr());
        cross_stream_addresses_.emplace_back(0, new_kernel_tensor->device_ptr());
      }
      temp_input_kernel_tensors_[i] = input_kernel_tensors_[i];
      input_kernel_tensors_[i] = new_kernel_tensor;
      input_launch_tensors_[i] = new_kernel_tensor.get();
    }
  }
}

void KernelActor::Run(OpContext<KernelTensor> *const context) {
  try {
    MS_EXCEPTION_IF_NULL(kernel_);
    MS_EXCEPTION_IF_NULL(kernel_->func_graph());
    if (NeedRunMemTracker()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), kernel_->fullname_with_scope(),
                                                     kernel_->func_graph()->ToString(), false);
    }
    FetchInputDeviceTensor(context);
    UpdateRefDeviceAddress(context, true);
    if (ActorDispatcher::enable_runtime_multi_pipeline()) {
      RunWithMultiPipeline(context);
      return;
    }

    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    InferAndUpdateDeviceTensorSize(context);

    // Set the memory address for the tensors which use the somas.
    SetSomasMemory(context);

    if (ActorDispatcher::enable_async_launch_kernel()) {
      RunWithAsyncLaunchKernel(context);
      return;
    }

    if (!memory_alloc_list_.empty()) {
      // Allocate the memory address for other tensors which don't use the somas.
      SendMemoryAllocReq(context);
    }
    OnMemoryAllocFinish(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }
}

void KernelActor::RunWithMultiPipeline(OpContext<KernelTensor> *const context) {
  // 1. Set the memory address for the tensors which use the somas if need.
  SetSomasMemory(context);

  // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this value
  // depend case can not handle in KernelTensor auto sync phase currently.
  if (kernel_mod_->need_user_data() && has_dynamic_) {
    MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
    if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }
    MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
  }

  // 2. Push run task to pipeline.
  // Note: dynamic value or static shape also need push task into infer actor to make sure correct kernel execution
  // order.
  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  Async(kernel_async_infer_aid_, &KernelAsyncInferActor::InferShape, context, this);

  // The computed depend kernel should wait output shape update after kernel launch.
  if (kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
    if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }
    MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
  }

  // 3. Post run.
  EraseInput(context);
  SendOutput(context);
}

void KernelActor::RunWithAsyncLaunchKernel(OpContext<KernelTensor> *const context) {
  Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, this);

  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  if (kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
    if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }
    MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_->fullname_with_scope();
  }

  // PostLaunchKernel
  EraseInput(context);
  SendOutput(context);
}

void KernelActor::FetchWorkspaceDeviceTensor() {
  auto workspace_sizes = kernel_mod_->GetWorkspaceSizeList();
  // Resize of workspace_kernel_tensors_, memory_alloc_list_ and memory_free_list_, because of
  // the dynamic size of workspace.
  if (workspace_kernel_tensors_.size() > workspace_sizes.size()) {
    size_t size = workspace_kernel_tensors_.size() - workspace_sizes.size();
    (void)workspace_kernel_tensors_.erase(workspace_kernel_tensors_.end() - size, workspace_kernel_tensors_.end());
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.workspaces_.erase(mem_info_.workspaces_.end() - size, mem_info_.workspaces_.end());
    }

    MS_EXCEPTION_IF_CHECK_FAIL((memory_alloc_list_.size() >= size), "The memory alloc list size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((memory_free_list_.size() >= size), "The memory free list size is wrong.");
    (void)memory_alloc_list_.erase(memory_alloc_list_.end() - size, memory_alloc_list_.end());
    (void)memory_free_list_.erase(memory_free_list_.end() - size, memory_free_list_.end());
    (void)new_memory_free_list_.erase(new_memory_free_list_.end() - size, new_memory_free_list_.end());
  } else if (workspace_kernel_tensors_.size() < workspace_sizes.size()) {
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      MS_LOG(ERROR) << "Invalid device context for kernel actor:" + GetAID().Name();
      return;
    }
    for (size_t i = workspace_kernel_tensors_.size(); i < workspace_sizes.size(); ++i) {
      auto kernel_tensor = AnfAlgo::CreateKernelTensor(
        nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
        device_contexts_[0]->device_context_key().device_name_, device_contexts_[0]->device_context_key().device_id_);
      kernel_tensor->set_stream_id(kernel_info_->stream_id());
      auto device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel_)
                    << " addr:" << device_address;
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel_);  // set to kernel_info
      (void)workspace_kernel_tensors_.emplace_back(kernel_tensor);
      if (recorder_aid_ != nullptr) {
        (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
      }
      (void)memory_alloc_list_.emplace_back(kernel_tensor);
      (void)memory_free_list_.emplace_back(kernel_tensor);
      (void)new_memory_free_list_.emplace_back(kernel_tensor);
    }
  }
  // Set workspace address new size
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto workspace_kernel_tensor = workspace_kernel_tensors_[i].get();
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensor);
    auto workspace_device_tensor = workspace_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(workspace_device_tensor);
    workspace_device_tensor->SetSize(workspace_sizes[i]);
  }

  // Update workspace kernel tensors.
  workspace_launch_tensors_.resize(workspace_kernel_tensors_.size());
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    workspace_launch_tensors_[i] = workspace_kernel_tensors_[i].get();
  }
}

void KernelActor::SetSomasMemory(OpContext<KernelTensor> *const context) const {
  if (!IsSomasEnable(somas_info_)) {
    return;
  }

  // Set the memory address for the output tensors which use the somas.
  const auto &somas_outputs = kernel_info_->somas_output_result();
  for (size_t i = 0; i < somas_outputs.size(); ++i) {
    if (somas_outputs[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_outputs[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      // Check the graph output address need free.
      auto output_device_tensor = output_kernel_tensors_[i]->device_address().get();
      MS_EXCEPTION_IF_NULL(output_device_tensor);
      if (somas_graph_output_indexes_.count(i) && (output_device_tensor->GetPtr() != nullptr)) {
        if (device_ptr != output_device_tensor->GetPtr()) {
          MS_LOG(ERROR) << GetAID().Name() << " does not free address for graph output index: " << i
                        << " device address:" << output_device_tensor->PrintInfo();
          device_contexts_[0]->device_res_manager_->FreeMemory(output_device_tensor);
        }
      }
      MS_LOG(DEBUG) << "Set ptr:" << device_ptr << " to device address:" << output_device_tensor
                    << " in actor:" << GetAID();
      output_device_tensor->set_ptr(device_ptr);
      if (somas_graph_output_indexes_.count(i) || output_device_tensor->new_ref_count() != SIZE_MAX) {
        output_device_tensor->IncreaseNewRefCount(GetAID().Name());
        MS_LOG(DEBUG) << "Add new ref count for somas output address:" << output_device_tensor
                      << " in kernel actor:" << GetAID();
      }
    }
  }

  // Set the memory address for the workspace tensors which use the somas.
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  for (size_t i = 0; i < somas_workspace.size(); ++i) {
    if (somas_workspace[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_workspace[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      auto &workspace_device_tensor = workspace_kernel_tensors_[i]->device_address();
      MS_EXCEPTION_IF_NULL(workspace_device_tensor);
      workspace_device_tensor->set_ptr(device_ptr);
    }
  }
}

void *KernelActor::GetSomasDevicePtr(size_t offset) const {
  // Get the ptr from the whole block.
  if (somas_info_->base_address_ != nullptr) {
    return AddressOffset(somas_info_->base_address_, offset);
  }

  // Get the ptr from the merged blocks.
  auto iter = somas_info_->merged_base_addresses_.upper_bound(offset);
  if (iter == somas_info_->merged_base_addresses_.begin()) {
    MS_LOG(ERROR) << GetAID().Name() << " can't find the merged block for offset: " << offset;
    return nullptr;
  }
  --iter;
  size_t real_offset = offset - iter->first;
  void *real_base_address = iter->second;
  if (real_base_address == nullptr) {
    MS_LOG(ERROR) << GetAID().Name() << " doesn't allocate the merged block base address for offset: " << iter->first;
    return nullptr;
  }
  return AddressOffset(real_base_address, real_offset);
}

void KernelActor::TraceDynamicMemory() {
  for (size_t i = 0; i < output_kernel_tensors_.size(); i++) {
    if (output_kernel_tensors_[i]->device_address()->original_ref_count() != SIZE_MAX) {
      const auto &kernel_tensor = output_kernel_tensors_[i];
      MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
        std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(),
                                                 kOutputMem, i, kernel_tensor.get()),
        device_contexts_[0]);
    }
  }

  for (size_t i = 0; i < workspace_kernel_tensors_.size(); i++) {
    const auto &kernel_tensor = workspace_kernel_tensors_[i];
    MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
      std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(),
                                               kWorkspaceMem, i, kernel_tensor.get()),
      device_contexts_[0]);
  }
}

void KernelActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  if (device_contexts_[0]->device_res_manager_->swap_manager() != nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_info_);
    for (const auto &out_in : kernel_info_->out_in_ref_map()) {
      const auto &input_kernel_tensor = input_kernel_tensors_[out_in.second];
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      const auto &input_device_tensor = input_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(input_device_tensor);
      const auto &ptr = input_device_tensor->GetValidPtr(kDefaultStreamIndex);
      MS_EXCEPTION_IF_NULL(output_kernel_tensors_[out_in.first]);
      const auto &output_device_tensor = output_kernel_tensors_[out_in.first]->device_address();
      if (ptr == nullptr || output_device_tensor == nullptr || output_device_tensor->GetPtr() != nullptr) {
        continue;
      }
      // Pointer in DeviceAddress which is reference output may not be updated to the same as the reference input
      // which is swapped out.
      MS_LOG(DEBUG) << "Set device ptr of " << out_in.first << "th ref output the same as input " << out_in.second
                    << ": " << ptr;
      output_device_tensor->set_ptr(ptr);
    }
  }
  MemoryManagerActor::GetInstance()->AllocateMemory(&memory_alloc_list_, device_contexts_[0], context, GetAID());

  if (ActorDispatcher::enable_trace_dynamic_memory()) {
    if (IsRunningFailed(context)) {
      return;
    }
    TraceDynamicMemory();
  }
}

void KernelActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MemoryManagerActor::GetInstance()->FreeMemory(&new_memory_free_list_, device_contexts_[0], context, GetAID());
  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_kernel_tensor : copy_input_kernel_tensors_) {
    if (copy_input_kernel_tensor == nullptr) {
      continue;
    }
    const auto &copy_input_device_tensor = copy_input_kernel_tensor->device_address();
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      MS_LOG(DEBUG) << "Free memory by ref count for device address:" << copy_input_device_tensor->PrintInfo()
                    << " for actor:" << GetAID();
      MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(copy_input_device_tensor.get(), device_contexts_[0],
                                                              GetAID().Name());
    }
  }
  // Free the address that is the temp store for kernel contiguous copy.
  for (auto &contiguous_kernel_tensor : contiguous_tensors_) {
    if (contiguous_kernel_tensor == nullptr) {
      continue;
    }
    auto &contiguous_device_tensor = contiguous_kernel_tensor->device_address();
    if ((contiguous_device_tensor != nullptr) && (contiguous_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(contiguous_device_tensor.get());
    }
  }
}

void KernelActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  PreLaunchKernel(context);

  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPreLaunch, kernel_, input_kernel_tensors_,
                              output_kernel_tensors_, device_contexts_[0], context, &GetAID());
  }

  bool skip_launch = CollectiveManager::instance()->need_reinit() || IsSkippedLaunch(kernel_, nullptr);
  if (!LaunchKernel(context, skip_launch)) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }
  // Record mem info, because async send may free device info.
  if (recorder_aid_ != nullptr) {
    SetMemInfoForRdr();
  }

  PostLaunchKernel(context);
}

void KernelActor::SetMemInfoForRdr() {
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (is_monad_input_[i]) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[i]);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[i]->device_address());
    mem_info_.inputs_[i]->addr = input_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.inputs_[i]->size = input_kernel_tensors_[i]->device_address()->GetSize();
  }
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_kernel_tensors_[i]->device_address());
    mem_info_.outputs_[i]->addr = output_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.outputs_[i]->size = output_kernel_tensors_[i]->device_address()->GetSize();
  }
  for (size_t i = 0; i < workspace_kernel_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensors_[i]->device_address());
    mem_info_.workspaces_[i]->addr = workspace_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.workspaces_[i]->size = workspace_kernel_tensors_[i]->device_address()->GetSize();
  }
}

void KernelActor::UpdateDeviceTensorCopyStore(DeviceTensor *const new_device_tensor,
                                              DeviceTensor *const input_device_tensor, size_t input_index) {
  UpdateRefCount(new_device_tensor, true);
  MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << new_device_tensor
                << " type:" << new_device_tensor->GetDeviceType() << " and " << input_device_tensor
                << " type:" << input_device_tensor->GetDeviceType() << " for copy actor:" << GetAID();
  DeviceTensorCopyStore::GetInstance().Insert(new_device_tensor, input_device_tensor);
}

void KernelActor::CopyInputDeviceTensor(KernelTensorPtr kernel_tensor, size_t input_index,
                                        OpContext<KernelTensor> *const context) {
  // The ignored input address that is not used in the kernel launch and no need copy.
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (!launch_ignored_inputs_.empty() && (std::find(launch_ignored_inputs_.begin(), launch_ignored_inputs_.end(),
                                                    input_index) != launch_ignored_inputs_.end())) {
    MS_LOG(DEBUG) << GetAID().Name() << " ignore the input address for input index: " << input_index;
    return;
  }
  if (skip_launch_shape_related_op_) {
    return;
  }
  if (input_index >= real_input_data_infos_.size()) {
    std::stringstream ofs;
    ofs << "Invalid input index:" << input_index << " size:" << real_input_data_infos_.size()
        << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
  }
  auto &real_input_info = real_input_data_infos_[input_index];
  if ((device_tensor->GetDeviceType() == device_contexts_[0]->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(device_tensor->format(), real_input_info->format_)) {
    return;
  }

  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  if (inputs_continuous_memory_) {
    std::string error_info = GetAID().Name() + " inputs must be continuous memory and can't be copied for index " +
                             std::to_string(input_index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (input_index >= copy_input_kernel_tensors_.size()) {
    std::stringstream ofs;
    ofs << "Invalid input index:" << input_index
        << " copy input device tensor size:" << copy_input_kernel_tensors_.size() << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
  }
  if (copy_input_kernel_tensors_[input_index] == nullptr) {
    const auto &pre_kernel_tensor = kernel_tensor;
    MS_EXCEPTION_IF_NULL(pre_kernel_tensor);
    auto new_kernel_tensor = AnfAlgo::CreateKernelTensor(
      pre_kernel_tensor->GetShape(), pre_kernel_tensor->GetType(), pre_kernel_tensor->GetValueTrack(), nullptr,
      real_input_info->size_, real_input_info->format_, real_input_info->type_id_, real_input_info->shape_,
      device_contexts_[0]->device_context_key().device_name_, device_contexts_[0]->device_context_key().device_id_,
      device_tensor->user_data());
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    auto pre_stream_id = pre_kernel_tensor->stream_id();
    if (pre_stream_id == UINT32_MAX) {
      auto stream_id = kernel_info_->stream_id();
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with kernel info stream id : " << stream_id << ".";
      new_kernel_tensor->set_stream_id(stream_id);
    } else {
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with pre kernel tensor stream id : " << pre_stream_id << ".";
      new_kernel_tensor->set_stream_id(pre_stream_id);
    }

    copy_input_kernel_tensors_[input_index] = new_kernel_tensor;
    MS_LOG(DEBUG) << "Create copy kernel tensor:" << copy_input_kernel_tensors_[input_index] << " index:" << input_index
                  << " for actor:" << GetAID();
  }
  auto &new_kernel_tensor = copy_input_kernel_tensors_[input_index];
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);
  auto &new_device_tensor = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  new_device_tensor->set_need_sync_user_data(device_tensor->need_sync_user_data());
  MS_LOG(DEBUG) << "Prev stream id : " << input_kernel_tensors_[input_index]->device_address()->stream_id()
                << " new stream id : " << new_device_tensor->stream_id() << ".";
  // Update the input kernel tensor.
  input_launch_tensors_[input_index] = new_kernel_tensor.get();
  pre_input_kernel_tensors_[input_index] = kernel_tensor;
  input_kernel_tensors_[input_index] = new_kernel_tensor;
  if (is_dynamic_shape_) {
    // Need update shape and size for dynamic shape case.
    input_kernel_tensors_for_infer_[input_index] = input_kernel_tensors_[input_index];
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[input_index]);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor->GetShape());
    input_kernel_tensors_[input_index]->SetShape(kernel_tensor->GetShape()->Clone());
    input_kernel_tensors_[input_index]->set_size(device_tensor->GetSize());
  }

  if (new_device_tensor->GetPtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                   new_device_tensor->GetSize(), new_device_tensor.get());
    if (!device_contexts_[0]->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *(device_contexts_[0]), GetAID().Name(),
                                                  new_device_tensor->GetSize());
    }
    MS_LOG(DEBUG) << "Increase new ref count for device address:" << new_device_tensor << " in actor:" << GetAID();
  }

  MS_LOG(INFO) << GetAID().Name() << " the input position:" << input_index
               << " copy from device address:" << device_tensor->PrintInfo()
               << " to device address:" << new_device_tensor->PrintInfo();
  // Copy from the real parameter to formal parameter and insert the device tensor copy store.
  if (!Copy(new_device_tensor.get(), device_tensor)) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (modifiable_ref_input_indexes_.count(input_index) > 0) {
    UpdateDeviceTensorCopyStore(new_device_tensor.get(), device_tensor, input_index);
  }
}

void KernelActor::UpdateInputDeviceTensor(const OpData<KernelTensor> *input_data,
                                          OpContext<KernelTensor> *const context) {
  size_t input_index = IntToSize(input_data->index_);
  if (input_index >= input_kernel_tensors_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
      strategy_, (*context),
      "The input index:" + std::to_string(input_index) + " is out of vector size:" +
        std::to_string(input_kernel_tensors_.size()) + " for kernel:" + kernel_->fullname_with_scope());
  }

  // Update the input kernel tensor.
  if (input_kernel_tensors_[input_index] != input_data->data_) {
    input_launch_tensors_[input_index] = input_data->data_.get();
    input_kernel_tensors_[input_index] = input_data->data_;
    memory_free_list_[input_index] = input_data->data_;
    if (is_dynamic_shape_) {
      input_kernel_tensors_for_infer_[input_index] = input_kernel_tensors_[input_index];
    }
  }
}

void KernelActor::FetchInputDeviceTensor(OpContext<KernelTensor> *const context) {
  // Collect the inputs from graph root parameter.
  if (enable_input_optimize_) {
    FetchParameterByTensorStore(&input_launch_tensors_, &input_kernel_tensors_, &input_kernel_tensors_for_infer_,
                                &memory_free_list_, context);
  }

  // Collect the inputs from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      UpdateInputDeviceTensor(input_data, context);
      CopyInputDeviceTensor(input_data->data_, IntToSize(input_data->index_), context);
    }
  }

  // Collect the inputs from device tensor store.
  FetchInputByTensorStore(&input_launch_tensors_, &input_kernel_tensors_, &input_kernel_tensors_for_infer_,
                          &memory_free_list_, context);

  // Collect the input free device tensor, when the pre input device tensor is not nullptr, it means the input device
  // tensor is heterogeneous and it should be freed in memory free list. And the real input will be freed by the copy
  // device tensors.
  for (size_t i = 0; i < input_free_index_.size(); ++i) {
    if (input_free_index_[i] >= input_kernel_tensors_.size() ||
        input_free_index_[i] >= pre_input_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        strategy_, (*context),
        "Invalid input index:" + std::to_string(input_free_index_[i]) +
          "] input size:" + std::to_string(input_kernel_tensors_.size()) + " pre input size:" +
          std::to_string(pre_input_kernel_tensors_.size()) + " for kernel:" + kernel_->fullname_with_scope());
    }
    new_memory_free_list_[i] =
      (pre_input_kernel_tensors_[input_free_index_[i]] == nullptr ? input_kernel_tensors_[input_free_index_[i]]
                                                                  : pre_input_kernel_tensors_[input_free_index_[i]]);
    MS_LOG(DEBUG) << "Add new memory free list for input index:" << input_free_index_[i]
                  << " input kernel tensor:" << input_kernel_tensors_[input_free_index_[i]]
                  << " and pre input kernel tensor:" << pre_input_kernel_tensors_[input_free_index_[i]]
                  << " for actor:" << GetAID();
    pre_input_kernel_tensors_[input_free_index_[i]] = nullptr;
  }
}

void KernelActor::UpdateGraphOutputRefCount(OpContext<KernelTensor> *const context) {
  for (const auto &pair : increase_ref_count_size_) {
    if (pair.first >= output_kernel_tensors_.size() || output_kernel_tensors_[pair.first] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << pair.first << " total size:" << output_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    const auto &output_device_tensor = output_kernel_tensors_[pair.first]->device_address();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    output_device_tensor->IncreaseNewRefCount(GetAID().Name(), pair.second);
    MS_LOG(DEBUG) << "Add new ref count size:" << pair.second
                  << " for kernel tensor:" << output_device_tensor->PrintInfo() << " for kernel actor:" << GetAID();
  }
}

void KernelActor::UpdateMemoryFreeList(OpContext<KernelTensor> *const context) {
  // Set input device address to memory free list by free index.
  for (size_t free_list_index = 0; free_list_index < input_free_index_.size(); ++free_list_index) {
    size_t input_list_index = input_free_index_[free_list_index];
    if (free_list_index >= new_memory_free_list_.size() || input_list_index >= input_kernel_tensors_.size() ||
        input_list_index >= pre_input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid free position:" << free_list_index
                        << " free list size:" << new_memory_free_list_.size() << " or input index:" << input_list_index
                        << " input size:" << input_kernel_tensors_.size()
                        << " pre input size:" << pre_input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_[free_list_index] =
      (pre_input_kernel_tensors_[input_list_index] == nullptr ? input_kernel_tensors_[input_list_index]
                                                              : pre_input_kernel_tensors_[input_list_index]);
    MS_LOG(DEBUG) << "Add new memory free list for input index:" << input_list_index
                  << " input kernel tensor:" << input_kernel_tensors_[input_list_index]
                  << " and pre input kernel tensor:" << pre_input_kernel_tensors_[input_list_index]
                  << " for kernel actor:" << GetAID();
    pre_input_kernel_tensors_[input_list_index] = nullptr;
  }
}

void KernelActor::UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  for (const auto &pair : kernel_info_->out_in_ref_map()) {
    if (pair.first >= output_kernel_tensors_.size() || pair.second >= input_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        strategy_, (*context),
        "Invalid ref index pair [" + std::to_string(pair.first) + ", " + std::to_string(pair.second) +
          "] input size:" + std::to_string(input_kernel_tensors_.size()) + " output size:" +
          std::to_string(output_kernel_tensors_.size()) + " for kernel:" + kernel_->fullname_with_scope());
    }
    if (output_kernel_tensors_[pair.first] == nullptr || input_kernel_tensors_[pair.second] == nullptr) {
      std::stringstream error_info;
      error_info << "Invalid ref input device address" << input_kernel_tensors_[pair.second]
                 << "and output kernel tensor:" << output_kernel_tensors_[pair.first]
                 << " for kernel:" + kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info.str());
    }
    auto input_device_tensor = input_kernel_tensors_[pair.second]->device_address().get();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto output_device_tensor = output_kernel_tensors_[pair.first]->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    output_device_tensor->set_pointer_ref_count(input_device_tensor->pointer_ref_count());
    output_device_tensor->IncreaseNewRefCount(GetAID().Name());

    MS_LOG(DEBUG) << "Actor:" << GetAID()
                  << " increase new ref count for device address:" << output_device_tensor->PrintInfo()
                  << " and input device address:" << input_device_tensor->PrintInfo();
  }
}

void KernelActor::FetchOutputDeviceTensor(OpContext<KernelTensor> *const context) {
  auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  const auto &output_size_list = kernel_mod_->GetOutputSizeList();

  // May exist in the kernel which does not support the dynamic shape.
  if (output_kernel_tensors.size() != output_size_list.size()) {
    std::string error_info = "The outputs number(" + std::to_string(output_size_list.size()) + ") is wrong, " +
                             GetAID().Name() + " may not support the dynamic shape, please check.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  // Update the size of output device tensor.
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_kernel_tensors[i]);
    auto &output_address = output_kernel_tensors[i]->device_address();
    if (output_size_list[i] == output_address->GetSize()) {
      continue;
    }
    output_address->SetSize(output_size_list[i]);
  }
}

void KernelActor::PreLaunchKernel(OpContext<KernelTensor> *) {
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto &input_device_tensor = input_kernel_tensors_[i]->device_address();
    if (input_device_tensor == nullptr || !input_device_tensor->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", input device tensor "
                    << input_device_tensor << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    if (output_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto &output_device_tensor = output_kernel_tensors_[i]->device_address();
    if (!output_device_tensor->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", output device tensor "
                    << output_device_tensor << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < workspace_kernel_tensors_.size(); ++i) {
    if (workspace_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto workspace_device_tensor = workspace_kernel_tensors_[i]->device_address().get();
    if (!workspace_device_tensor->GetValidPtr(kernel_info_->stream_id())) {
      MS_LOG(DEBUG) << "For kernel: " << kernel_->fullname_with_scope() << ", workspace device tensor "
                    << workspace_device_tensor << " has no device ptr.";
    }
  }
}

void KernelActor::ExecuteInferShapeTask(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInfer, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop infer shape for kernel: " << kernel_->fullname_with_scope();
    return;
  }

  if (is_dynamic_type_) {
    InferShapeAndType();
  } else if (is_dynamic_shape_) {
    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    InferShape();
  }

  Async(kernel_async_resize_aid_, &KernelAsyncResizeActor::ResizeKernelMod, context, this);
}

void KernelActor::ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop resize for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  bool view_input = false;
  if (!need_check_tensor_contiguous_) {
    auto it = std::find_if(input_kernel_tensors_.begin(), input_kernel_tensors_.end(),
                           [](const KernelTensorPtr &tensor) { return tensor->tensor_storage_info() != nullptr; });
    if (it != input_kernel_tensors_.end()) {
      view_input = true;
    }
  }

  if (has_dynamic_ || view_input) {
    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    ResizeKernelMod();

    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else {
    FetchOutputDeviceTensor(context);
  }

  Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, this);
}

void KernelActor::ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) {
  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop launch kernel: " << kernel_->fullname_with_scope();
    return;
  }
  // 1. Allocate memory.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (!memory_alloc_list_.empty()) {
      SendMemoryAllocReq(context);
    }
  } else if (!max_ref_cnt_output_list_.empty()) {
    // Allocate dynamic memory for graph output.
    MemoryManagerActor::GetInstance()->AllocateMemory(&max_ref_cnt_output_list_, device_contexts_[0], context,
                                                      GetAID());
  }

  if (IsRunningFailed(context)) {
    MS_LOG(DEBUG) << "Run failed and early stop launch kernel: " << kernel_->fullname_with_scope();
    return;
  }
  // For performance, Only kernel need user data (such as PyExecute op) need call 'PreLaunchKernel', the
  // 'PreLaunchKernel' will be removed in the future.
  if (ActorDispatcher::has_kernel_need_user_data()) {
    PreLaunchKernel(context);
  }

  // 2. Launch kernel if need.
  device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);

  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPreLaunch, kernel_, input_kernel_tensors_,
                              output_kernel_tensors_, device_contexts_[0], context, &GetAID());
  }

  if (!LaunchKernel(context, IsSkippedLaunch(kernel_, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }

  if (recorder_aid_ != nullptr) {
    SetMemInfoForRdr();
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                          device_contexts_[0], context);
  }

  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_launch_tensors_, output_launch_tensors_);
  }

  if (kernel_mod_->need_user_data()) {
    for_each(output_kernel_tensors_.begin(), output_kernel_tensors_.end(),
             [](auto &kernel_tensor) { kernel_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // 3. Fix ref count.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    IncreaseNewRefCounts(context);
    if (new_memory_free_list_.size() > 0 && copy_output_kernel_tensors_.empty()) {
      SendMemoryFreeReq(context);
    }
  }
}

void KernelActor::InferAndUpdateDeviceTensorSize(OpContext<KernelTensor> *const context) {
  // For static shape, aclnn kernel with view input need to get input tensor storage info by resize.
  bool view_input = false;
  if (!need_check_tensor_contiguous_) {
    auto it = std::find_if(input_kernel_tensors_.begin(), input_kernel_tensors_.end(),
                           [](const KernelTensorPtr &tensor) { return tensor->tensor_storage_info() != nullptr; });
    if (it != input_kernel_tensors_.end()) {
      view_input = true;
    }
  }
  if (has_dynamic_) {
    // Infer shape and resize for dynamic shape or dynamice value case when disable runtime multi pipeline.
    InferAndResize(context);
    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else if (view_input) {
    ResizeKernelMod();
    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else {
    FetchOutputDeviceTensor(context);
  }
}

void KernelActor::InferAndResize(OpContext<KernelTensor> *const context) {
  if (!enable_async_infer_) {
    // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this
    // value depend case can not handle in KernelTensor auto sync phase currently.
    if (ActorDispatcher::enable_async_launch_kernel() && kernel_mod_->need_user_data() &&
        !WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }

    if (is_dynamic_type_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShapeAndType();
      ResizeKernelMod();
    } else if (is_dynamic_shape_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShape();
      ResizeKernelMod();
    } else if (is_dynamic_value_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
      ResizeKernelMod();
    }

    return;
  }

  if (is_dynamic_value_ && !is_dynamic_shape_ && !is_dynamic_type_) {
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
    ResizeKernelMod();
  }
}

void KernelActor::InferShapeAndType() {
  MS_LOG(DEBUG) << "Begin InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape and Type.
  auto abstract = opt::dynamic_shape::InferShapeAndType(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(abstract);
  MS_LOG(DEBUG) << "End InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                << ", abstract: " << abstract->ToString();
  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorType(abstract->GetType(), output_launch_tensors_);
  opt::dynamic_shape::UpdateKernelTensorShape(abstract->GetShape(), output_launch_tensors_);
}

void KernelActor::InferShape() {
  MS_LOG(DEBUG) << "Begin InferShape for kernel: " << kernel_->fullname_with_scope()
                << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape.
  auto base_shape = opt::dynamic_shape::InferShape(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_LOG(DEBUG) << "End InferShape for kernel: " << kernel_->fullname_with_scope()
                << ", shape: " << base_shape->ToString();

  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorShape(base_shape, output_launch_tensors_);
}

void KernelActor::ResizeKernelMod() {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResizeInner, GetAID().Name(), true);
  MS_LOG(DEBUG) << "Begin Resize kernel mod for kernel: " << kernel_->fullname_with_scope();
  int ret = kernel_mod_->Resize(input_launch_tensors_, output_launch_tensors_);
  MS_LOG(DEBUG) << "End Resize kernel mod for kernel: " << kernel_->fullname_with_scope()
                << ", the output size list: " << kernel_mod_->GetOutputSizeList()
                << ", workspace size list: " << kernel_mod_->GetWorkspaceSizeList();
  if (ret != kernel::KRET_OK) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "Resize failed for kernel: " << kernel_->fullname_with_scope();
  }
}

void KernelActor::DispatchDebugActor(OpContext<KernelTensor> *const context) {
  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPostLaunch, kernel_, input_kernel_tensors_,
                              output_kernel_tensors_, device_contexts_[0], context, &GetAID());
  }
}

bool KernelActor::LaunchKernelWithDebug(OpContext<KernelTensor> *const context, const bool skip_launch) {
  MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();
  static bool is_enable_mem_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (is_enable_mem_tracker) {
    AddNodeToGraphTracker(kernel_, GetAID().Name());
    TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(), depend_shape_input_list_);
  } else {
    if (device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info()) {
      AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
    }
  }
  bool ret = true;
  if (!skip_launch) {
    ret = device_contexts_[0]->GetKernelExecutor(false)->LaunchKernel(
      kernel_, input_launch_tensors_, workspace_launch_tensors_, output_launch_tensors_, kernel_mod_, stream_);
  }
  MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();
  DispatchDebugActor(context);
  return ret;
}

void KernelActor::RecoverInputs() {
  if (!temp_input_kernel_tensors_.empty()) {
    for (const auto &pair : temp_input_kernel_tensors_) {
      input_kernel_tensors_[pair.first] = pair.second;
      input_launch_tensors_[pair.first] = pair.second.get();
    }
    temp_input_kernel_tensors_.clear();
  }
}

bool KernelActor::LaunchKernel(OpContext<KernelTensor> *const context, bool is_skip_launch) {
  static KernelCache &cache = KernelCache::GetInstance();
  if (cache.need_add) {
    cache.Add(kernel_);
  }

  if (EnableExecuteOrderDump()) {
    auto &execute_order_tracker = ExecuteOrderTracker::GetInstance();
    execute_order_tracker.ProcessNode(kernel_);
  }
  static bool is_enable_mem_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (skip_launch_shape_related_op_) {
    MS_LOG(DEBUG) << "Skip launch real make tuple kernel: " << kernel_->fullname_with_scope()
                  << " input kernel tensor: " << input_kernel_tensors_;
    if (is_enable_mem_tracker) {
      AddNodeToGraphTracker(kernel_, GetAID().Name());
      TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(), depend_shape_input_list_);
    } else {
      if (device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info()) {
        AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
      }
    }
    return true;
  }
  // Check the skipped launch condition.
  if (is_launch_skipped_) {
    MS_EXCEPTION_IF_CHECK_FAIL((input_kernel_tensors_.size() >= 1), "The inputs size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((output_kernel_tensors_.size() >= 1), "The outputs size is wrong.");
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
    MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]);
    auto &input_device_tensor = input_kernel_tensors_[0]->device_address();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto &output_device_tensor = output_kernel_tensors_[0]->device_address();
    if (input_device_tensor->GetPtr() == output_device_tensor->GetPtr()) {
      MS_LOG(DEBUG) << "Skipped launch kernel: " << kernel_->fullname_with_scope();
      DispatchDebugActor(context);
      if (is_enable_mem_tracker) {
        AddNodeToGraphTracker(kernel_, GetAID().Name());
        TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(),
                               depend_shape_input_list_);
      } else {
        if (device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info()) {
          AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
        }
      }
      return true;
    } else {
      MS_LOG(ERROR) << "Input address:" << input_device_tensor->GetPtr()
                    << " and output address:" << output_device_tensor->GetPtr()
                    << " are not equal of skipped launch actor: " << GetAID().Name();
      return false;
    }
  }
  // Make tensor contiguous if needed
  if (need_check_tensor_contiguous_) {
    ConvertInputContiguous(context);
  }

  // Cpu not support stream lock with LaunchKernel.
  if (!ActorDispatcher::enable_multi_stream() || is_multi_stream_process_skipped_) {
    auto ret = LaunchKernelWithDebug(context, is_skip_launch);
    RecoverInputs();
    return ret;
  }

  auto &multi_stream_controller =
    device::HalResManager::GetInstance().GetMultiStreamController(device_contexts_[0]->DeviceName());
  bool ret = false;
  if (!ActorDispatcher::enable_async_launch_kernel()) {
    std::lock_guard<std::mutex> lock(multi_stream_controller->GetStreamMutex(kernel_info_->stream_id()));
    ProcessMultiStreamBeforeKernelLaunch(context);
    ret = LaunchKernelWithDebug(context, is_skip_launch);
    ProcessMultiStreamAfterKernelLaunch(context);
  } else {
    ProcessMultiStreamBeforeKernelLaunch(context);
    ret = LaunchKernelWithDebug(context, is_skip_launch);
    ProcessMultiStreamAfterKernelLaunch(context);
  }
  RecoverInputs();
  return ret;
}

void KernelActor::ProcessMultiStreamBeforeKernelLaunch(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kProcessMultiStream, GetAID().Name());
  auto device_context = device_contexts_[0];
  auto stream_id = kernel_info_->stream_id();
  // Update output_kernel_tensors_ with task id on stream.
  auto &multi_stream_controller =
    device::HalResManager::GetInstance().GetMultiStreamController(device_context->DeviceName());
  auto task_id_on_stream = multi_stream_controller->LaunchTaskIdOnStream(stream_id);
  // Adapter for mc2 kernel, need more process later.
  if (is_mc2_kernel_) {
    multi_stream_controller->DispatchRecordWaitEvent(kDefaultStreamIndex, kWorldGroupStreamIndex);
  }
  MS_LOG(DEBUG) << "device context : " << device_context
                << ", name : " << device_context->device_context_key().device_name_ << ", stream id : " << stream_id
                << ", actor name : " << GetAID().Name() << ", task_id_on_stream : " << task_id_on_stream << ".";
  if (INT64_MAX == task_id_on_stream) {
    // Cpu kernel task id on stream is meanless.
    *task_id_on_stream_ = 0;
    MS_LOG(DEBUG) << "Skip ProcessMultiStreamBeforeKernelLaunch since kernel type is CPU.";
    return;
  }
  *task_id_on_stream_ = task_id_on_stream;

  // Process wait stream.
  if (is_stream_recv_actor_) {
    // Note: wait node start to launch. Event was record on send node, so, we can releases events on send node stream.
    // Release events on send node means memory stream id is recv node stream id and user stream id is send node
    // stream id.
    auto user_stream_id = kernel_mod_->record_stream_id();
    auto memory_stream_id = stream_id;
    if (stream_send_actor_ == nullptr) {
      // Gpu not add stream send/recv pair, nullptr is normal case.
      MS_LOG(DEBUG) << "Stream_send_actor_ is nullptr.";
      return;
    }
    MS_LOG(DEBUG) << "Process wait stream start, memory_stream_id : " << memory_stream_id
                  << ", send task id on stream : " << *(stream_send_actor_->task_id_on_stream_) << ".";
    // Here, need get task id on stream from send node.
    (void)multi_stream_controller->WaitEvent(*(stream_send_actor_->task_id_on_stream_), user_stream_id,
                                             memory_stream_id);
    return;
  }

  // Process inputs.
  if (input_kernel_tensors_.empty()) {
    return;
  }

  std::vector<KernelTensor *> cross_stream_kernel_tensors;
  size_t index = 0;
  for (const auto &input_kernel_tensor : input_kernel_tensors_) {
    if (is_monad_input_[index++]) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    if (input_kernel_tensor->stream_id() == stream_id) {
      continue;
    }
    if (input_kernel_tensor->task_id_on_stream() == nullptr) {
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor
                    << " task id on stream is nullptr, will skip multi stream process.";
      continue;
    }
    if (input_kernel_tensor->managed_by_somas()) {
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor << " is managed by somas.";
      continue;
    }
    // Nullptr device ptr is normal case, here need skip these inputs.
    if (input_kernel_tensor->device_ptr() == nullptr) {
      MS_LOG(DEBUG) << "Input kernel tensor device ptr is nullptr.";
      continue;
    }
    (void)cross_stream_addresses_.emplace_back(kDefaultStreamIndex, input_kernel_tensor->device_ptr());
    if (!is_multi_stream_safe_) {
      (void)cross_stream_kernel_tensors.emplace_back(input_kernel_tensor.get());
    }
  }

  // Dispatch record/wait.
  if (!is_multi_stream_safe_) {
    for (const auto &cross_stream_kernel_tensor : cross_stream_kernel_tensors) {
      // Nullptr of task id on stream is normal case.
      // If cross_stream_kernel_tensor's task id on stream is nullptr, kernel tensor must be safe.
      // Data prepare actor, data source actor and so on has prepare device tensors without task id on stream, and
      // those device tensors is multi-stream safe.
      if (cross_stream_kernel_tensor->task_id_on_stream() == nullptr) {
        continue;
      }
      // Input kernel tensor is memory stream id, this is important.
      auto user_stream_id = stream_id;
      auto memory_stream_id = cross_stream_kernel_tensor->stream_id();
      auto memory_task_id_on_stream = *cross_stream_kernel_tensor->task_id_on_stream();
      auto safe_task_id_on_stream = multi_stream_controller->QueryTaskIdOnStream(user_stream_id, memory_stream_id);
      if (safe_task_id_on_stream >= memory_task_id_on_stream) {
        MS_LOG(DEBUG) << "Safe_task_id_on_stream : " << safe_task_id_on_stream
                      << " is bigger than memory_task_id_on_stream : " << memory_task_id_on_stream << ".";
        continue;
      }
      multi_stream_controller->DispatchRecordWaitEvent(user_stream_id, memory_stream_id);
      // Add recv process.
      user_stream_id = memory_stream_id;
      memory_stream_id = stream_id;
      auto last_task_id_on_stream = multi_stream_controller->GetTaskIdOnStream(user_stream_id);
      MS_LOG(DEBUG) << "Dispatch wait stream start, user_stream_id : " << user_stream_id
                    << ", memory_stream_id : " << memory_stream_id
                    << ", last_task_id_on_stream : " << last_task_id_on_stream << ".";
      // Here, need get task id on stream from send node.
      (void)multi_stream_controller->WaitEvent(last_task_id_on_stream, user_stream_id, memory_stream_id);
    }
  }
}

void KernelActor::ProcessMultiStreamAfterKernelLaunch(OpContext<KernelTensor> *const context) {
  auto stream_id = kernel_info_->stream_id();
  if (stream_id != kDefaultStreamIndex) {
    for (const auto &workspace_kernel_tensor : workspace_kernel_tensors_) {
      cross_stream_addresses_.emplace_back(kDefaultStreamIndex, workspace_kernel_tensor->device_ptr());
    }
    for (const auto &input_kernel_tensor : input_kernel_tensors_) {
      if (input_kernel_tensor->stream_id() == stream_id) {
        cross_stream_addresses_.emplace_back(kDefaultStreamIndex, input_kernel_tensor->device_ptr());
      }
    }
    for (const auto &output_kernel_tensor : output_kernel_tensors_) {
      cross_stream_addresses_.emplace_back(kDefaultStreamIndex, output_kernel_tensor->device_ptr());
    }

    // Record event.
    if (!cross_stream_addresses_.empty()) {
      MS_LOG(DEBUG) << "Record event for kernel : " << kernel_->fullname_with_scope()
                    << ", addresses size : " << cross_stream_addresses_.size() << ".";
      // Record event on stream.
      auto device_context = device_contexts_[0];
      auto &multi_stream_controller =
        device::HalResManager::GetInstance().GetMultiStreamController(device_context->DeviceName());
      multi_stream_controller->RecordEvent(*task_id_on_stream_, stream_id, cross_stream_addresses_);
    }
  }
  // Reset cross stream addresses.
  cross_stream_addresses_.clear();
}

void KernelActor::PostLaunchKernel(OpContext<KernelTensor> *const context) {
  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_launch_tensors_, output_launch_tensors_);
  }

  if (kernel_mod_->need_user_data()) {
    for_each(output_kernel_tensors_.begin(), output_kernel_tensors_.end(),
             [](auto &device_tensor) { device_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // The input is invalid and needs to be erased when finish kernel launch.
  EraseInput(context);

  IncreaseNewRefCounts(context);
  // Note that SendMemoryFreeReq must be in front of SendOutput, because SendOutput will trigger SendMemoryAllocReq
  // of the next actor and the actor is asynchronous execution. So it is necessary to ensure that SendMemoryFreeReq
  // of the current actor is in front of SendMemoryAllocReq of the next actor. One is to reuse the memory more
  // fully, the other is to ensure the execution order and avoid the illegal memory timing problem.
  if (new_memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }

  SendOutput(context);
}

void KernelActor::RefreshDeviceTensorCopyStore(OpContext<KernelTensor> *const context) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  for (auto &ref_input_index : modifiable_ref_input_indexes_) {
    if (ref_input_index >= input_kernel_tensors_.size()) {
      std::stringstream ofs;
      ofs << "Invalid ref input index:" << ref_input_index
          << " input device tensor size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
    }
    auto &input_kernel_tensor = input_kernel_tensors_[ref_input_index];
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    auto input_device_tensor = input_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(input_device_tensor);
    MS_LOG(DEBUG) << "Fetch input copy device tensor:" << input_device_tensor << " for actor:" << GetAID();
    if (need_refreshed_device_tensors == nullptr) {
      continue;
    }
    for (auto &new_device_tensor : *need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << ref_input_index
                   << " refresh from device address:" << input_device_tensor->PrintInfo()
                   << " to device address:" << new_device_tensor->PrintInfo();

      if (new_device_tensor->GetPtr() == nullptr || new_device_tensor->GetPtr() == input_device_tensor->GetPtr()) {
        continue;
      }

      if (!Copy(new_device_tensor, input_device_tensor)) {
        std::string error_info = "Copy input device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  for (auto &ref_output_index : modifiable_ref_output_indexes_) {
    if (ref_output_index >= output_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The output index is of range.");
    }
    auto &output_kernel_tensor = output_kernel_tensors_[ref_output_index];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_device_tensor = output_kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(output_device_tensor);
    MS_LOG(DEBUG) << "Fetch output copy device tensor:" << output_device_tensor << " for actor:" << GetAID();
    if (need_refreshed_device_tensors == nullptr) {
      continue;
    }
    for (auto &new_device_tensor : *need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the output position:" << ref_output_index
                   << " refresh from device address:" << output_device_tensor
                   << " ptr:" << output_device_tensor->GetPtr() << ", type:" << output_device_tensor->GetDeviceType()
                   << ", format:" << output_device_tensor->format() << " to device address:" << new_device_tensor
                   << " ptr:" << new_device_tensor->GetPtr() << ", type:" << new_device_tensor->GetDeviceType()
                   << ", format:" << new_device_tensor->format();
      if (new_device_tensor->GetPtr() == nullptr || new_device_tensor->GetPtr() == output_device_tensor->GetPtr()) {
        continue;
      }

      if (!Copy(new_device_tensor, output_device_tensor)) {
        std::string error_info = "Copy output device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, GetAID().Name(), false);
}

void KernelActor::SendRecorderInfo(OpContext<KernelTensor> *const context) const {
  if (recorder_aid_ != nullptr && !ActorDispatcher::enable_async_launch_kernel()) {
    MS_EXCEPTION_IF_NULL(kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                          device_contexts_[0], context);
  }
}

void KernelActor::SetInputDeviceTensor(const KernelTensorPtr &input_kernel_tensor, size_t input_index) {
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  input_launch_tensors_[input_index] = input_kernel_tensor.get();
  input_kernel_tensors_[input_index] = input_kernel_tensor;
  input_kernel_tensors_for_infer_[input_index] = input_kernel_tensor;
}

void KernelActor::ResetState() {
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(INFO) << "Kernel actor " << kernel_->fullname_with_scope() << " start to reset state.";
  auto device_context = const_cast<DeviceContext *>(device_contexts_[0]);
  MS_LOG(INFO) << "Free output_device_tensor, list size: " << output_kernel_tensors_.size();
  for (auto kernel_tensor : output_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      continue;
    }
    auto device_tensor = kernel_tensor->device_address();
    if (device_tensor->new_ref_count() == SIZE_MAX) {
      continue;
    }
    if (device_tensor != nullptr && device_tensor->GetPtr() != nullptr) {
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (held_by_nodes.empty()) {
        FreeMemoryByDeviceContext(kernel_tensor->device_address().get(), device_context);
      } else {
        FreeMemoryByValueNode(held_by_nodes, device_tensor.get());
      }
    }
  }
  MS_LOG(INFO) << "Free workspace_device_tensor, list size: " << workspace_kernel_tensors_.size();
  for (auto kernel_tensor : workspace_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      continue;
    }
    auto device_tensor = kernel_tensor->device_address();
    if (device_tensor != nullptr && device_tensor->GetPtr() != nullptr) {
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (held_by_nodes.empty()) {
        FreeMemoryByDeviceContext(kernel_tensor->device_address().get(), device_context);
      } else {
        FreeMemoryByValueNode(held_by_nodes, device_tensor.get());
      }
    }
  }
  MS_LOG(INFO) << "Kernel actor " << kernel_->fullname_with_scope() << " end to reset state.";
}
}  // namespace runtime
}  // namespace mindspore
