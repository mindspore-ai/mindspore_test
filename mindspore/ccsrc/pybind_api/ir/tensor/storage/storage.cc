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
#include "pybind_api/ir/tensor/storage/storage.h"
#include <utility>
#include <string>
#include "include/utils/pybind_api/api_register.h"

namespace mindspore {
Storage::~Storage() { storage_base_ = nullptr; }

uintptr_t Storage::DataPtr() const { return storage_base_->DataPtr(); }

void Storage::InplaceReSize(int64_t size) { return storage_base_->InplaceReSize(size); }

int64_t Storage::NBytes() const { return storage_base_->NBytes(); }

void Storage::InplaceCopy(const Storage &src, bool non_blocking) {
  return storage_base_->InplaceCopy(src.storage_base_, non_blocking);
}

std::string Storage::device() const { return storage_base_->device(); }
}  // namespace mindspore
