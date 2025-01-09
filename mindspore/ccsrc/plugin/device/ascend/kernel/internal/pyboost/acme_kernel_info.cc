/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/pyboost/acme_kernel_info.h"

#include <functional>
#include <utility>
#include "plugin/device/ascend/kernel/internal/pyboost/acme_pyboost_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_helper.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "transform/acl_ir/op_api_cache.h"
#include "kernel/common/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
void AcmeKernelInfo::UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor) {
  arginfo->SetDtype(TransInternalDataType(tensor->data_type()));
  auto device_sync = tensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  arginfo->SetFormat(TransInternalFormat(device_address->GetFormat()));
}

void AcmeKernelInfo::UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> &arginfos,
                                            const std::vector<BaseTensorPtr> &tensorlist) {
  arginfos.resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); ++i) {
    UpdateArgImmutableInfo(&(arginfos[i]), tensorlist[i]);
  }
}

bool AcmeKernelInfo::Init(const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs) {
  internal::InputsImmutableInfoList inputs_ii;
  internal::OutputsImmutableInfoList outputs_ii;
  UpdateArgImmutableInfo(inputs_ii, inputs);
  UpdateArgImmutableInfo(outputs_ii, outputs);
  acme_op_ = CreateKernel(inputs_ii, outputs_ii);
  MS_EXCEPTION_IF_NULL(acme_op_);

  auto status = acme_op_->Init();
  if (status != internal::kInternalOk) {
    acme_op_ = nullptr;
    MS_LOG(ERROR) << "Init AcmeKernel failed, kenrel_name: " << kernel_name_;
    return false;
  }

  acme_inputs_shape_.resize(inputs.size());
  acme_inputs_addr_.resize(inputs.size());
  acme_outputs_shape_.resize(outputs.size());
  acme_outputs_addr_.resize(outputs.size());

  workspace_size_list_ = acme_op_->GetWorkspaceSize();
  acme_wss_addr_.resize(workspace_size_list_.size());

  return true;
}

void AcmeKernelInfo::TransAcmeShapes(internal::ShapeInfoList &shapelist, const std::vector<BaseTensorPtr> &tensorlist) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    auto shape =
      tensorlist[i]->data_type() != kMetaTypeNone ? TransInternalShape(tensorlist[i]->shape()) : internal::ShapeInfo{};
    shapelist[i] = std::move(shape);
  }
}

TilingCacheItemPtr AcmeKernelInfo::GetOrGenerateTiling(const std::vector<BaseTensorPtr> &inputs,
                                                       const std::vector<BaseTensorPtr> &outputs) {
  TransAcmeShapes(acme_inputs_shape_, inputs);
  TransAcmeShapes(acme_outputs_shape_, outputs);
  auto acme_ret = acme_op_->UpdateShape(acme_inputs_shape_, acme_outputs_shape_);
  if (acme_ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "AcmeKernel UpdateShape failed, kernel_name: " << kernel_name_;
    return nullptr;
  }

  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto key = CalcAcmeOpTilingHash(kernel_name_, inputs);
  auto tiling_info_ptr = InternalTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    auto tiling_size = acme_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    internal::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = acme_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != internal::kInternalOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << kernel_name_ << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }

    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<internal::TilingInfo>(device_addr, nullptr);
    tiling_info->host_run_info_ = host_run_info_ptr;
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list_);
    tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    if (TilingMemMgr::GetInstance().pool_device_.IsOneOffMem(device_addr)) {
      // tiling mem pool is full, comb out some items which are not recently used with high probability
      auto erased_items = InternalTilingCache::GetInstance().CombOutSuspectedUselessItems();
      if (!erased_items.empty()) {
        for (auto &item : erased_items) {
          TilingMemMgr::GetInstance().pool_device_.Free(item->tiling_info_->tiling_addr_, item->size_);
          TilingMemMgr::GetInstance().pool_host_.Free(item->host_addr_, item->size_);
        }
        TilingMemMgr::GetInstance().pool_device_.Rearrange();
        TilingMemMgr::GetInstance().pool_host_.Rearrange();
      }
      MS_LOG(INFO) << "The tiling memory pool is full, comb out not used items: " << erased_items.size();
    }
    (void)InternalTilingCache::GetInstance().Insert(key, tiling_info_ptr);
  }

  return tiling_info_ptr;
}

void AcmeKernelInfo::UpdateAddr(std::vector<internal::RawDeviceAddr> &addrlist,
                                const std::vector<BaseTensorPtr> &tensorlist) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    addrlist[i] = tensorlist[i]->device_address()->GetMutablePtr();
  }
}

void AcmeKernelInfo::MallocWorkspace(const device::DeviceContext *device_context, size_t stream_id) {
  for (size_t i = 0; i < workspace_size_list_.size(); i++) {
    auto ptr = device_context->device_res_manager_->AllocateMemory(workspace_size_list_[i], stream_id);
    if (ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Alloc failed, size:" << workspace_size_list_[i] << ", stream_id:" << stream_id;
    }
    acme_wss_addr_[i] = ptr;
  }
}

void AcmeKernelInfo::FreeWorkspace(const device::DeviceContext *device_context) {
  for (size_t i = 0; i < acme_wss_addr_.size(); i++) {
    device_context->device_res_manager_->FreeMemory(acme_wss_addr_[i]);
    acme_wss_addr_[i] = nullptr;
  }
}

void AcmeKernelInfo::Launch(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<BaseTensorPtr> &inputs,
                            const TilingCacheItemPtr tilingptr) {
  pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([this, op, inputs, tilingptr]() {
    MS_LOG(DEBUG) << "Launch AcmeKernel " << kernel_name_ << "start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    pyboost::PyBoostUtils::MallocOpInputs(device_context, inputs);
    // Malloc for output tensors
    pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs);
    UpdateAddr(acme_inputs_addr_, inputs);
    UpdateAddr(acme_outputs_addr_, outputs);
    runtime::Pipeline::Get().launch_stage()->Wait();

    MallocWorkspace(device_context, op->stream_id());
    acme_op_->SetTilingInfo(tilingptr->tiling_info_);
    auto stream_ptr = device_context->device_res_manager_->GetStream(op->stream_id());
    internal::InternalStatus status =
      acme_op_->Launch(acme_inputs_addr_, acme_outputs_addr_, acme_wss_addr_, stream_ptr, kernel_name_);
    FreeWorkspace(device_context);
    InternalTilingCache::GetInstance().Unbind(tilingptr);
    if (status != internal::InternalStatus::kInternalOk) {
      MS_LOG(EXCEPTION) << "Launch AcmeKernel failed, kernel_name: " << kernel_name_;
    }
    MS_LOG(DEBUG) << "Launch AcmeKernel " << kernel_name_ << "end";
  }));
}

void AcmeKernelInfo::CallAcmeOp(const std::shared_ptr<pyboost::OpRunner> &op, const std::vector<BaseTensorPtr> &inputs,
                                uint64_t key) {
  auto it = hash_map_.find(key);
  if (it != hash_map_.end()) {
    acme_op_ = it->second;
  } else {
    auto ret = Init(inputs, op->outputs());
    if (!ret) {
      return;
    }
    hash_map_[key] = acme_op_;
  }
  auto tiling_info = GetOrGenerateTiling(inputs, op->outputs());
  if (tiling_info == nullptr) {
    return;
  }
  Launch(op, inputs, tiling_info);
}

}  // namespace kernel
}  // namespace mindspore
