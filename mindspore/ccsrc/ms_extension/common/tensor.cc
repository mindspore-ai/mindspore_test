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
#include <algorithm>
#include <functional>
#include "ir/tensor.h"
#include "mindspore/ccsrc/include/common/utils/tensor_utils.h"
#include "mindspore/ccsrc/pynative/pynative_utils.h"
#include "mindspore/ccsrc/pipeline/jit/ps/parse/data_converter.h"
#include "frontend/ir/tensor_py.h"
#include "include/common/utils/stub_tensor.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"

namespace ms {
namespace {
inline mindspore::Int64ImmPtr MakeI64Value(int64_t v) { return std::make_shared<mindspore::Int64Imm>(v); }
inline mindspore::ValueTuplePtr MakeI64Tuple(const std::vector<int64_t> &v) {
  return mindspore::MakeValue(v)->cast<mindspore::ValueTuplePtr>();
}
inline std::vector<Tensor> ToTensorList(const std::vector<mindspore::tensor::TensorPtr> &tensors) {
  std::vector<Tensor> outs;
  outs.reserve(tensors.size());
  (void)std::transform(tensors.begin(), tensors.end(), std::back_inserter(outs), [](auto &t) { return Tensor(t); });
  return outs;
}
}  // namespace

Tensor::RealTensorHolder::RealTensorHolder(const mindspore::ValuePtr &value)
    : value_(value), tensor_(value->cast<mindspore::tensor::TensorPtr>()) {}

Tensor::Tensor(TypeId type_id, const ShapeVector &shape)
    : Tensor(std::make_shared<mindspore::tensor::Tensor>(type_id, shape)) {}

Tensor::Tensor(const mindspore::ValuePtr &value) {
  if (value != nullptr) {
    _tensor_holder_ = std::make_shared<RealTensorHolder>(value);
  }
}

void *Tensor::GetDataPtr() const {
  auto t = tensor();
  MS_EXCEPTION_IF_NULL(t);
  if (t->device_address() == nullptr) {
    return nullptr;
  }
  int64_t offset = static_cast<int64_t>(t->data().itemsize()) * t->storage_offset();
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

Tensor Tensor::cast(TypeId dtype) const {
  return Tensor(mindspore::kernel::pyboost::cast(tensor(), MakeI64Value(static_cast<int64_t>(dtype))));
}

std::vector<Tensor> Tensor::chunk(int64_t chunks, int64_t dim) const {
  return ToTensorList(mindspore::kernel::pyboost::chunk(tensor(), MakeI64Value(chunks), MakeI64Value(dim)));
}

Tensor Tensor::contiguous() const { return Tensor(mindspore::kernel::pyboost::contiguous(tensor())); }

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
  return Tensor(mindspore::kernel::pyboost::flatten_ext(tensor(), MakeI64Value(start_dim), MakeI64Value(end_dim)));
}

Tensor Tensor::index_select(int64_t dim, const Tensor &index) const {
  return Tensor(mindspore::kernel::pyboost::index_select(tensor(), MakeI64Value(dim), index.tensor()));
}

Tensor Tensor::reshape(const std::vector<int64_t> &shape) const {
  return Tensor(mindspore::kernel::pyboost::reshape(tensor(), MakeI64Tuple(shape)));
}

Tensor Tensor::repeat(const std::vector<int64_t> &repeats) const {
  return Tensor(mindspore::kernel::pyboost::repeat(tensor(), MakeI64Tuple(repeats)));
}

Tensor Tensor::repeat_interleave(const Tensor &repeats, const std::optional<int64_t> &dim,
                                 const std::optional<int64_t> &output_size) const {
  std::optional<mindspore::Int64ImmPtr> dim_opt = std::nullopt;
  std::optional<mindspore::Int64ImmPtr> output_size_opt = std::nullopt;
  if (dim.has_value()) {
    dim_opt = MakeI64Value(dim.value());
  }
  if (output_size.has_value()) {
    output_size_opt = MakeI64Value(output_size.value());
  }
  return Tensor(
    mindspore::kernel::pyboost::repeat_interleave_tensor(tensor(), repeats.tensor(), dim_opt, output_size_opt));
}

Tensor Tensor::repeat_interleave(int64_t repeats, const std::optional<int64_t> &dim,
                                 const std::optional<int64_t> &output_size) const {
  std::optional<mindspore::Int64ImmPtr> dim_opt = std::nullopt;
  std::optional<mindspore::Int64ImmPtr> output_size_opt = std::nullopt;
  if (dim.has_value()) {
    dim_opt = MakeI64Value(dim.value());
  }
  if (output_size.has_value()) {
    output_size_opt = MakeI64Value(output_size.value());
  }
  return Tensor(
    mindspore::kernel::pyboost::repeat_interleave_int(tensor(), MakeI64Value(repeats), dim_opt, output_size_opt));
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
