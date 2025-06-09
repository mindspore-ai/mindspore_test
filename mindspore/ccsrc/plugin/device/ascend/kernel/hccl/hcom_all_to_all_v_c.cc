/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_all_to_all_v_c.h"

namespace mindspore {
namespace kernel {
namespace {
void Tranpose1d(int64_t rank_size_int64, std::vector<int64_t> &list_1d) {
  // if (rank_size * rank_size != size) {
  //   MS_LOG(INTERNAL_EXCEPTION) << "The size of the one-dimensional array cannot form a square matrix.";
  // }
  // int64_t output_numel = 0;
  //for (int64_t i = 0; i < rank_size; ++i) {
  //  for (int64_t j = 0; j < rank_size; ++j) {
  //    auto temp = list_1d[i * rank_size + j];
  //    list_1d[i * rank_size + j] = list_1d[j * rank_size + i];
  //    list_1d[j * rank_size + i] = temp;
  //  }
  //}
  size_t rank_size = LongToSize(rank_size_int64);
  for (size_t i = 0; i < rank_size; ++i) {
    for (size_t j = i + 1; j < rank_size; ++j) {
        size_t original_index = i * rank_size + j;
        size_t transposed_index = j * rank_size + i;
        auto temp = list_1d[original_index];
        list_1d[original_index] = list_1d[transposed_index];
        list_1d[transposed_index] = temp;
    }
  }
}
}  // namespace
int64_t HcomAlltoAllVCKernel::GetOutputNumel(int64_t block_size, const std::vector<int64_t> &list_1d) {
  int64_t size = list_1d.size();
  if (rank_size_ * rank_size_ != size) {
    MS_LOG(INTERNAL_EXCEPTION) << "The size of the one-dimensional array cannot form a square matrix.";
  }
  params_ = std::shared_ptr<int64_t[]>(new int64_t[size]);
  int64_t output_numel = 0;
  for (int64_t i = 0; i < rank_size_; ++i) {
    for (int64_t j = 0; j < rank_size_; ++j) {
      params_[i * rank_size_ + j] = list_1d[i * rank_size_ + j] * block_size;
      if (rank_id_ == j) {
        output_numel += list_1d[i * rank_size_ + j] * block_size;
      }
    }
  }

  return output_numel;
}

bool HcomAlltoAllVCKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
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

int HcomAlltoAllVCKernel::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  output_size_list_.clear();
  std::vector<int64_t> send_count_matrix;
  if (inputs.size() == kInputNum2) {
    send_count_matrix = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid hccl AlltoAllVC input size " << inputs.size();
  }
  auto block_size = GetValue<int64_t>(primitive_->GetAttr(kAttrBlockSize));
  auto transpose = GetValue<bool>(primitive_->GetAttr(kAttrTransPose));
  if (transpose) {
    Tranpose1d(rank_size_, send_count_matrix);
  }
  int64_t output_numel = GetOutputNumel(block_size, send_count_matrix);
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

bool HcomAlltoAllVCKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
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
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAlltoAllVC(
    send_buf, reinterpret_cast<void *>(params_.get()), data_type_, recv_buf, data_type_, stream_ptr, comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAlltoAllVC failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
