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

#include "ms_extension/common/tensor.h"
#include <functional>
#include "ir/tensor.h"
#include "ir/tensor_api.h"
#include "mindspore/ccsrc/include/common/utils/tensor_utils.h"
#include "mindspore/ccsrc/pynative/pynative_utils.h"
#include "mindspore/ccsrc/pipeline/jit/ps/parse/data_converter.h"
#include "frontend/ir/tensor_py.h"
#include "include/common/utils/stub_tensor.h"

namespace ms {
Tensor::RealTensorHolder::RealTensorHolder(const mindspore::ValuePtr &value)
    : value_(value), tensor_(value->cast<mindspore::tensor::TensorPtr>()) {}

Tensor::Tensor(TypeId type_id, const ShapeVector &shape)
    : Tensor(mindspore::tensor::empty(type_id, shape, mindspore::device::DeviceType::kNone)) {}

Tensor::Tensor(const mindspore::ValuePtr &value) {
  if (value != nullptr) {
    _tensor_holder_ = std::make_shared<RealTensorHolder>(value);
  }
}

void *Tensor::GetDataPtr() const {
  auto t = tensor();
  MS_EXCEPTION_IF_NULL(t);
  auto device_address = t->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  int64_t offset = static_cast<int64_t>(t->DataItemSize()) * t->storage_offset();
  return static_cast<void *>(static_cast<int8_t *>(t->device_address()->GetMutablePtr()) + offset);
}

TypeId Tensor::data_type() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->data_type();
}

const ShapeVector &Tensor::shape() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->shape();
}

size_t Tensor::numel() const {
  auto &s = shape();
  return static_cast<size_t>(std::accumulate(s.begin(), s.end(), 1LL, std::multiplies<int64_t>()));
}

std::vector<int64_t> Tensor::stride() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->stride();
}

int64_t Tensor::storage_offset() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->storage_offset();
}

bool Tensor::is_contiguous() const {
  MS_EXCEPTION_IF_NULL(tensor());
  return tensor()->is_contiguous();
}

void Tensor::SetNeedContiguous(bool flag) const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  _tensor_holder_->need_contiguous_ = flag;
}

bool Tensor::need_contiguous() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->need_contiguous_;
}

const mindspore::ValuePtr &Tensor::stub_node() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->value_;
}

const mindspore::tensor::TensorPtr &Tensor::tensor() const {
  MS_EXCEPTION_IF_CHECK_FAIL(is_defined(), "The Tensor is not defined");
  return _tensor_holder_->tensor_;
}

void Tensor::ConvertStubNodeToTensor() const {
  if (stub_node() != nullptr) {
    _tensor_holder_->tensor_ =
      mindspore::pynative::PyNativeAlgo::Common::ConvertStubNodeToTensor(stub_node(), need_contiguous(), false);
    // release the stub node object.
    _tensor_holder_->value_ = nullptr;
  }
}
}  // namespace ms

namespace pybind11 {
namespace detail {
bool type_caster<ms::Tensor>::load(handle src, bool) {
  if (mindspore::tensor::IsTensorPy(src)) {
    auto v = mindspore::tensor::ConvertToValue(src);
    if (v->isa<mindspore::tensor::Tensor>()) {
      v->cast<mindspore::tensor::TensorPtr>()->set_need_pipeline_sync(true);
    }
    value = ms::Tensor(v);
    return true;
  }
  // the value is initialized as an undefined Tensor for None input.
  return src.is_none();
}

handle type_caster<ms::Tensor>::cast(const ms::Tensor &src, return_value_policy, handle) {
  MS_EXCEPTION_IF_NULL(src.tensor());
  return handle(mindspore::tensor::Wrap(src.tensor()));
}
}  // namespace detail
}  // namespace pybind11
