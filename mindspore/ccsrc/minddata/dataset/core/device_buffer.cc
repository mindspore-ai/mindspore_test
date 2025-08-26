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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/device_buffer.h"

namespace mindspore {
namespace dataset {
DeviceBuffer::DeviceBuffer(const std::vector<size_t> &shape)
    : shape_(shape), ptr_(nullptr), own_data_(true), device_context_(nullptr) {
  if (shape.size() != 0) {
    strides_ = std::vector<size_t>(shape.size());
    size_ = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides_[i] = size_;
      size_ *= shape[i];
    }
  } else {
    size_ = 0;
  }

  if (size_ != 0) {
    auto ms_context = MsContext::GetInstance();
    if (ms_context == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Get ms context failed by MsContext::GetInstance().";
    }
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET)),
       ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    if (device_context_ == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Get device context failed by ms context.";
    }
    device_context_->Initialize();
    if (device_context_->device_res_manager_ == nullptr) {
      MS_EXCEPTION(RuntimeError) << "The device resource manager is null.";
    }
    ptr_ = device_context_->device_res_manager_->AllocateMemory(size_);
    if (ptr_ == nullptr) {
      MS_EXCEPTION(RuntimeError) << "Allocate device memory failed.";
    }
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other)
    : shape_(std::move(other.shape_)),
      size_(other.size_),
      ptr_(std::move(other.ptr_)),
      own_data_(other.own_data_),
      strides_(std::move(other.strides_)),
      device_context_(std::move(other.device_context_)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) {
  if (&other != this) {
    shape_ = std::move(other.shape_);
    ptr_ = std::move(other.ptr_);
    size_ = other.size_;
    own_data_ = other.own_data_;
    strides_ = std::move(other.strides_);
    device_context_ = std::move(other.device_context_);
  }
  return *this;
}

DeviceBuffer::DeviceBuffer(const std::shared_ptr<DeviceBuffer> &other, ptrdiff_t offset) {
  if (other->shape_.size() == 0 || offset >= other->shape_[0]) {
    MS_EXCEPTION(RuntimeError) << "Offset " << offset << " is larger than dim size";
  }
  shape_ = other->shape_;
  (void)shape_.erase(shape_.begin());
  size_ = other->size_ / other->shape_[0];
  ptr_ = reinterpret_cast<void *>(reinterpret_cast<char *>(other->ptr_) + offset * other->strides_[0]);
  own_data_ = false;
  strides_ = other->strides_;
  (void)strides_.erase(strides_.begin());
  device_context_ = nullptr;
}

DeviceBuffer::~DeviceBuffer() {
  if (own_data_ && ptr_ != nullptr && device_context_ != nullptr && device_context_->device_res_manager_ != nullptr) {
    device_context_->device_res_manager_->FreeMemory(ptr_);
    ptr_ = nullptr;
  }
}

size_t DeviceBuffer::GetBufferSize() const { return size_; }

void *DeviceBuffer::GetBuffer() { return ptr_; }

std::vector<size_t> DeviceBuffer::GetShape() { return shape_; }

std::vector<size_t> DeviceBuffer::GetStrides() { return strides_; }
}  // namespace dataset
}  // namespace mindspore
