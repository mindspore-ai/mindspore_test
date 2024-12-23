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
#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "transform/acl_ir/op_api_cache.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
void AcmeKernelInfo::UpdateArgImmutableInfo(acme::ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor) {
  arginfo->SetDtype(TransAcmeDataType(tensor->data_type()));
  auto device_sync = tensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  auto format = TransAcmeFormat(device_address->GetFormat());
  arginfo->SetFormat(format);
}

void AcmeKernelInfo::UpdateArgImmutableInfo(const std::vector<BaseTensorPtr> &tensorlist,
                                            std::vector<acme::ArgImmutableInfo> &arginfos) {
  arginfos.resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); i++) {
    UpdateArgImmutableInfo(&(arginfos[i]), tensorlist[i]);
  }
}

bool AcmeKernelInfo::Init(const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs) {
  acme::InputsImmutableInfoList inputs_ii;
  acme::OutputsImmutableInfoList outputs_ii;
  UpdateArgImmutableInfo(inputs, inputs_ii);
  UpdateArgImmutableInfo(outputs, outputs_ii);

  acme_op_ = CreateKernel(inputs_ii, outputs_ii, inputs, outputs);
  MS_EXCEPTION_IF_NULL(acme_op_);
  auto status = acme_op_->Init();
  if (status != acme::kAcmeOk) {
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

void AcmeKernelInfo::TransAcmeShapes(const std::vector<BaseTensorPtr> &tensorlist, acme::ShapeInfoList &shapelist) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    auto shape =
      tensorlist[i]->data_type() != kMetaTypeNone ? TransAcmeShape(tensorlist[i]->shape()) : acme::ShapeInfo{};
    shapelist[i] = std::move(shape);
  }
}

TilingCacheItemPtr AcmeKernelInfo::GetOrGenerateTiling(const std::vector<BaseTensorPtr> &inputs,
                                                       const std::vector<BaseTensorPtr> &outputs) {
  TransAcmeShapes(inputs, acme_inputs_shape_);
  TransAcmeShapes(outputs, acme_outputs_shape_);

  auto acme_ret = acme_op_->UpdateShape(acme_inputs_shape_, acme_outputs_shape_);
  if (acme_ret != acme::kAcmeOk) {
    MS_LOG(ERROR) << "AcmeKernel UpdateShape failed, kernel_name: " << kernel_name_;
    return nullptr;
  }

  auto key = CalcAcmeOpTilingHash(kernel_name_, inputs);
  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto tiling_info_ptr = AcmeTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    auto tiling_size = acme_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    acme::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = acme_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != acme::kAcmeOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << kernel_name_ << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }

    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<acme::TilingInfo>(device_addr, nullptr);
    tiling_info->host_run_info_ = host_run_info_ptr;
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list_);
    tiling_info_ptr = std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    if (TilingMemMgr::GetInstance().pool_device_.IsOutOfPoolMem(device_addr)) {
      // tiling mem pool is full, comb out some items which are not recently used with high probability
      auto erased_items = AcmeTilingCache::GetInstance().CombOutSuspectedUselessItems();
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
    (void)AcmeTilingCache::GetInstance().Insert(key, tiling_info_ptr);
  }

  return tiling_info_ptr;
}

void AcmeKernelInfo::UpdateAddr(const std::vector<BaseTensorPtr> &tensorlist, acme::InputsAddrList &addrlist) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    auto device_sync = tensorlist[i]->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    addrlist[i] = device_address->GetMutablePtr();
  }
}

void AcmeKernelInfo::UpdateAddr(const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs) {
  UpdateAddr(inputs, acme_inputs_addr_);
  UpdateAddr(outputs, acme_outputs_addr_);
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

bool AcmeKernelInfo::Launch(const device::DeviceContext *device_context, const TilingCacheItemPtr tilingptr,
                            const std::vector<BaseTensorPtr> &inputs, const std::vector<BaseTensorPtr> &outputs,
                            size_t stream_id) {
  UpdateAddr(inputs, outputs);
  MallocWorkspace(device_context, stream_id);
  acme_op_->SetTilingInfo(tilingptr->tiling_info_);
  auto stream_ptr = device_context->device_res_manager_->GetStream(stream_id);
  acme::AcmeStatus status =
    acme_op_->Launch(acme_inputs_addr_, acme_outputs_addr_, acme_wss_addr_, stream_ptr, kernel_name_);
  FreeWorkspace(device_context);
  AcmeTilingCache::GetInstance().Unbind(tilingptr);
  return (status == acme::AcmeStatus::kAcmeOk);
}

void AcmeKernelInfo::Call(const OpRunnerPtr &op, const std::vector<BaseTensorPtr> &inputs,
                          const TilingCacheItemPtr tilingptr) {
  // Async
  pyboost::PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([this, op, inputs, tilingptr]() {
    MS_LOG(DEBUG) << "Run device task Add start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    pyboost::PyBoostUtils::MallocOpInputs(device_context, inputs);
    // Malloc for output tensors
    pyboost::PyBoostUtils::MallocOpOutputs(device_context, outputs);

    runtime::Pipeline::Get().launch_stage()->Wait();
    Launch(device_context, tilingptr, inputs, outputs, op->stream_id());
    MS_LOG(DEBUG) << "Run device task Add end";
  }));
}
}  // namespace kernel
}  // namespace mindspore
