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

#include "kernel/ascend/hccl/hcom_all_gather_v.h"

namespace mindspore {
namespace kernel {
bool HcomAllGatherVKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  std::vector<KernelTensor *> temp;
  temp.push_back(inputs[0]);
  if (!HcclKernel::Init(temp, outputs)) {
    MS_LOG(ERROR) << "HcclKernel Init failed.";
    return false;
  }
  rank_id_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("rank_id")));
  rank_size_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("rank_size")));
  data_type_ = HcomUtil::ConvertHcclType(inputs[0]->dtype_id());
  return true;
}

uint64_t HcomAllGatherVKernel::GetAllGatherVParam(uint64_t send_count, const std::vector<int64_t> &output_split_sizes) {
  params_.send_count = send_count;
  params_.recv_counts.clear();
  params_.rdispls.clear();
  uint64_t offset = 0;
  for (size_t i = 0; i < output_split_sizes.size(); i++) {
    auto count = LongToSize(output_split_sizes[i]);
    params_.recv_counts.push_back(count);
    params_.rdispls.push_back(offset);
    offset += count;
  }

  return offset;
}

int HcomAllGatherVKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  std::vector<int64_t> output_split_sizes;
  if (inputs.size() == kInputNum2) {
    output_split_sizes = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "AllGatherV input size must be 2, but got " << inputs.size();
  }
  auto input_shape = inputs[0]->GetDeviceShapeVector();
  auto send_count = SizeOf(input_shape);

  uint64_t output_numel = GetAllGatherVParam(send_count, output_split_sizes);
  ShapeVector shape;
  if (output_numel != 0) {
    shape.push_back(output_numel);
  }

  size_t size = 0;
  if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), shape, &size)) {
    MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
  }
  if (!outputs.empty()) {
    output_size_list_.push_back(size);
  }
  return KRET_OK;
}

bool HcomAllGatherVKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto send_tensor = inputs[0];
  MS_EXCEPTION_IF_NULL(send_tensor);
  auto send_buf = send_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(send_buf);
  void *recv_buf = nullptr;
  if (!outputs.empty()) {  // may be empty output when AlltoAllV is from NeighborExchangeV2
    auto recv_tensor = outputs[0];
    MS_EXCEPTION_IF_NULL(recv_tensor);
    recv_buf = recv_tensor->device_ptr();
    MS_EXCEPTION_IF_NULL(recv_buf);
  }
  auto hccl_result =
    hccl::HcclAdapter::GetInstance().HcclAllGatherV(send_buf, recv_buf, params_, data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllGatherV failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
