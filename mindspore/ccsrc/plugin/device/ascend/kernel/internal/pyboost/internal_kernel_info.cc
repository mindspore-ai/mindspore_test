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

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

#include <functional>
#include <utility>
#include "transform/acl_ir/op_api_cache.h"

namespace mindspore {
namespace kernel {
void InternalKernelInfo::GetInputAndOutputIndex(const std::shared_ptr<pyboost::OpRunner> &op,
                                                const ValuePtrList &input_values) {
  bool is_mutable = false;
  auto input_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelInMap(kernel_name_, &is_mutable);
  if (is_mutable) {
    for (size_t i = 0; i < input_values.size(); i++) {
      (void)ms_inputs_idx_list_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < input_idx_list.size(); i++) {
      auto ms_index = input_idx_list[i];
      (void)ms_inputs_idx_list_.emplace_back(static_cast<size_t>(ms_index));
    }
  }
  is_mutable = false;
  auto output_idx_list = InternalKernelModInOutMap::GetInstance()->GetKernelOutMap(kernel_name_, &is_mutable);
  if (is_mutable) {
    for (size_t i = 0; i < op->outputs().size(); i++) {
      (void)ms_outputs_idx_list_.emplace_back(i);
    }
  } else {
    for (size_t i = 0; i < output_idx_list.size(); i++) {
      auto ms_index = output_idx_list[i];
      (void)ms_outputs_idx_list_.emplace_back(static_cast<size_t>(ms_index));
    }
  }
}

void InternalKernelInfo::TransInternalShapes(internal::ShapeInfoList &shapelist,
                                             const std::vector<BaseTensorPtr> &tensorlist) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    if (tensorlist[i] == nullptr) {
      continue;
    }
    auto shape =
      tensorlist[i]->data_type() != kMetaTypeNone ? TransInternalShape(tensorlist[i]->shape()) : internal::ShapeInfo{};
    shapelist[i] = std::move(shape);
  }
}

void InternalKernelInfo::UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const BaseTensorPtr &tensor) {
  if (tensor == nullptr) {
    arginfo->SetDtype(internal::DataType::kTypeNone);
    arginfo->SetFormat(internal::TensorFormat::kFormatND);
    return;
  }
  arginfo->SetDtype(TransInternalDataType(tensor->data_type()));
  auto device_sync = tensor->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  arginfo->SetFormat(TransInternalFormat(device_address->GetFormat()));
}

void InternalKernelInfo::UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> &arginfos,
                                                const std::vector<BaseTensorPtr> &tensorlist) {
  arginfos.resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); ++i) {
    UpdateArgImmutableInfo(&(arginfos[i]), tensorlist[i]);
  }
}

bool InternalKernelInfo::Init(const ValuePtrList &input_values, std::vector<BaseTensorPtr> &inputs,
                              std::vector<BaseTensorPtr> &outputs, const std::vector<BaseTensorPtr> &op_outputs) {
  for (size_t i = 0; i < ms_inputs_idx_list_.size(); i++) {
    auto ms_index = ms_inputs_idx_list_[i];
    auto input_value = input_values[ms_index];
    if (input_value->isa<None>()) {
      (void)inputs.emplace_back(nullptr);
    } else {
      (void)inputs.emplace_back(input_value->cast<BaseTensorPtr>());
    }
  }
  for (size_t i = 0; i < ms_outputs_idx_list_.size(); i++) {
    auto ms_index = ms_outputs_idx_list_[i];
    (void)outputs.emplace_back(op_outputs[ms_index]);
  }

  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(internal_inputs_shape_, inputs);
  TransInternalShapes(internal_outputs_shape_, outputs);
  return true;
}

void InternalKernelInfo::GetOrCreateKernel(const std::vector<BaseTensorPtr> &inputs,
                                           const std::vector<BaseTensorPtr> &outputs, uint64_t key) {
  auto it = hash_map_.find(key);
  if (it != hash_map_.end()) {
    internal_op_ = it->second;
  } else {
    UpdateArgImmutableInfo(inputs_ii_, inputs);
    UpdateArgImmutableInfo(outputs_ii_, outputs);
    internal_op_ = CreateKernel(inputs_ii_, outputs_ii_);
    MS_EXCEPTION_IF_NULL(internal_op_);
    auto status = internal_op_->Init();
    if (status != internal::kInternalOk) {
      internal_op_ = nullptr;
      MS_LOG(ERROR) << "Init internal kernel failed, kenrel_name: " << kernel_name_;
      return;
    }
    hash_map_[key] = internal_op_;
  }
  tiling_info_ = GetOrGenerateTiling(inputs);
  if (tiling_info_ == nullptr) {
    MS_LOG(ERROR) << "Create tiling info failed for internal kernel, kernel_name: " << kernel_name_;
  }
}

TilingCacheItemPtr InternalKernelInfo::GetOrGenerateTiling(const std::vector<BaseTensorPtr> &inputs) {
  auto internal_ret = internal_op_->UpdateShape(internal_inputs_shape_, internal_outputs_shape_);
  if (internal_ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "InternalKernel UpdateShape failed, kernel_name: " << kernel_name_;
    return nullptr;
  }

  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto key = CalcInternalOpTilingHash(kernel_name_, inputs);
  auto tiling_info_ptr = InternalTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    auto tiling_size = internal_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    internal::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = internal_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != internal::kInternalOk || host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << kernel_name_ << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }

    auto device_addr = TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info = std::make_shared<internal::TilingInfo>(device_addr, nullptr);
    tiling_info->host_run_info_ = host_run_info_ptr;
    auto workspace_size_list = internal_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list);
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

}  // namespace kernel
}  // namespace mindspore
