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

#include "backend/ge_backend/runtime/actor/abstract_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {

std::atomic<int64_t> gActorId = 0;

AbstractActor::AbstractActor(const std::string &name, KernelTransformType type, const AID *recorder_aid)
    : OpRTActor(name),
      type_(type),
      recorder_aid_(recorder_aid),
      actor_id_(++gActorId),
      input_datas_num_(0),
      input_controls_num_(0),
      running_dependent_msg_num_(0) {}

void AbstractActor::RunOpData(OpData<KernelTensor> *const input_data, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  // The unused data may be invalid ptr.
  const auto &device_tensor = input_data->data_->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (!input_data->data_->IsPtrValid() && (!TEST_FLAG(input_data->data_->flag(), device::kDeviceAddressFlagNotUsed) &&
                                           !TEST_FLAG(input_data->data_->flag(), device::kDeviceAddressFlagNullptr))) {
    std::string error_info = "The input_data does not have a valid ptr of actor:" + GetAID().Name() +
                             " with index:" + std::to_string(input_data->index_) +
                             ", flag:" + std::to_string(input_data->data_->flag()) +
                             " device address:" + std::to_string((int64_t)(device_tensor.get())) +
                             " ref count:" + std::to_string(input_data->data_->ref_count()) +
                             " dynamic ref count:" + std::to_string(input_data->data_->dynamic_ref_count()) +
                             " origin ref count:" + std::to_string(input_data->data_->original_ref_count());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << is_run
                << ", sequential num:" << sequential_num << ", the input data:" << input_data->data_
                << " input index:" << input_data->index_ << ", size:" << input_data->data_->GetSize()
                << " ptr:" << input_data->data_->device_address()->GetMutablePtr()
                << ", origin ref count:" << input_data->data_->original_ref_count()
                << ", current ref count:" << input_data->data_->ref_count()
                << ", dynamic ref count:" << input_data->data_->dynamic_ref_count()
                << ", flag:" << input_data->data_->flag() << " user data:" << input_data->data_->user_data()
                << " from memory pool:" << input_data->data_->device_address()->from_mem_pool()
                << " device type:" << input_data->data_->GetDeviceType();
  if (is_run) {
    Run(context);
  }
}

void AbstractActor::RunOpControl(AID *const input_control, OpContext<KernelTensor> *const context) {
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);

  auto is_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name()
                << ") receive the input op control from:" << (input_control == nullptr ? "null" : input_control->Name())
                << " and check running condition:" << is_run << ", sequential num:" << sequential_num;
  if (is_run) {
    Run(context);
  }
}

bool AbstractActor::CheckRunningCondition(const OpContext<KernelTensor> *context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() < input_datas_num_) {
      return false;
    } else if (data_iter->second.size() > input_datas_num_) {
      MS_LOG(ERROR) << "Invalid input data num:" << data_iter->second.size() << " need:" << input_datas_num_
                    << " for actor:" << GetAID() << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    const auto &control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() < input_controls_num_) {
      return false;
    } else if (control_iter->second.size() > input_controls_num_) {
      MS_LOG(ERROR) << "Invalid input control num:" << control_iter->second.size() << " need:" << input_controls_num_
                    << " for actor:" << GetAID() << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }
  return true;
}

void AbstractActor::EraseInput(const OpContext<KernelTensor> *context) {
  (void)input_op_datas_.erase(context->sequential_num_);
  (void)input_op_controls_.erase(context->sequential_num_);
}

void AbstractActor::FetchInputByTensorStore(
  std::vector<KernelTensor *> *const input_launch_tensors, std::vector<KernelTensorPtr> *const input_kernel_tensors,
  std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
  std::vector<KernelTensorPtr> *const memory_free_tensors, OpContext<KernelTensor> *const context) const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_type = device::GetDeviceTypeByName(context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    const auto &kernel_tensor =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(), device_type);
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      std::string error_info = GetAID().Name() +
                               " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
                               ", device type:" + std::to_string(static_cast<int>(device_type));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    // Collect the input kernel tensor.
    if (input_launch_tensors && input_kernel_tensors && input_kernel_tensors_for_infer &&
        ((*input_kernel_tensors)[device_tensor_store_key.first] != kernel_tensor)) {
      (*input_launch_tensors)[device_tensor_store_key.first] = kernel_tensor.get();
      (*input_kernel_tensors)[device_tensor_store_key.first] = kernel_tensor;
      (*input_kernel_tensors_for_infer)[device_tensor_store_key.first] = kernel_tensor;
      (*memory_free_tensors)[device_tensor_store_key.first] = kernel_tensor;
    }
  }
}

void AbstractActor::InitOutputData() {
  mindspore::HashMap<std::string, size_t> batch_op_count;
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    auto data = std::make_unique<OpData<KernelTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    auto &to_op_name = data_arrow->to_op_id_.Name();

    // Identify whether the output data flag is kOutputDataFlagToStack.
    bool is_to_stack = (to_op_name.find(kStackActorNameSuffix) != std::string::npos);
    size_t output_data_flag = is_to_stack ? kOutputDataFlagToStack : kOutputDataFlagInit;

    // Add the output data.
    (void)output_data_.emplace_back(std::make_pair(std::move(data), output_data_flag));
  }
}

void AbstractActor::SendOutputData(
  OpContext<KernelTensor> *const context, const std::vector<AnfNodePtr> &output_data_nodes,
  const std::vector<DataArrowPtr> &output_data_arrows,
  const std::vector<std::pair<OpDataUniquePtr<KernelTensor>, size_t>> &output_data_list) {
  for (size_t i = 0; i < output_data_list.size(); ++i) {
    auto &output_data = output_data_list[i];
    auto &to_op_id = output_data.first->op_id_;
    auto &output_data_arrow = output_data_arrows[i];
    UpdateOutputData(output_data.first.get(), output_data_arrow, output_data_nodes[i], context);

    if (TEST_FLAG(output_data.second, kOutputDataFlagToStack)) {
      // Create a new op data for stack actor.
      auto to_stack_data =
        std::make_unique<OpData<KernelTensor>>(to_op_id, output_data.first->data_, output_data.first->index_);
      (void)to_stack_data_.emplace_back(std::move(to_stack_data));
      ActorDispatcher::Send(to_op_id, &OpRTActor::RunOpData, to_stack_data_.back().get(), context);
    } else {
      ActorDispatcher::Send(to_op_id, &OpRTActor::RunOpData, output_data.first.get(), context);
    }
  }
}

void AbstractActor::SendOutput(OpContext<KernelTensor> *const context) {
  // Must be the execution order: send data --> send control, avoid the illegal timing problem.
  // 1.Send output data.
  SendOutputData(context, output_data_nodes_, output_data_arrows_, output_data_);

  // 2.Send output control.
  if (output_control_arrows_.size() > 0) {
    auto from_aid = const_cast<AID *>(&GetAID());
    for (auto &output_control : output_control_arrows_) {
      ActorDispatcher::Send(output_control->to_op_id_, &OpRTActor::RunOpControl, from_aid, context);
    }
  }

  // 3.Send recorder info.
  SendRecorderInfo(context);
}

bool AbstractActor::IsOutputAddressPersisted(const KernelTensorPtr &output_kernel_tensor,
                                             const KernelWithIndex &output_node) {
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  MS_EXCEPTION_IF_NULL(output_node.first);
  // The persisted address can't be replaced.
  if (output_kernel_tensor->is_ptr_persisted()) {
    return true;
  }

  if (output_node.first->isa<ValueNode>()) {
    return true;
  }

  // The device address of parameter may come from the device address of input tensor.
  // In order to avoid mistakenly cleaning up the device data of input tensor, return it as persisted address.
  if (output_node.first->isa<Parameter>()) {
    return true;
  }

  // Ref node need check the origin node.
  const auto &graph = AnfAlgo::FetchKernelGraph(output_node.first.get());
  if ((graph != nullptr) && graph->IsInRefOutputMap(output_node)) {
    const auto &origin_node = graph->GetRefNodeRecursive(output_node).first;
    MS_EXCEPTION_IF_NULL(origin_node);
    if (origin_node->isa<ValueNode>() || origin_node->isa<Parameter>()) {
      return true;
    }
  }

  return false;
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
