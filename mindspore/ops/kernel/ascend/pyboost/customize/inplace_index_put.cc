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

#include <functional>
#include "kernel/ascend/pyboost/customize/inplace_index_put.h"
#include "kernel/ascend/pyboost/auto_generate/inner_non_zero.h"
#include "kernel/ascend/pyboost/auto_generate/select_ext.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/common/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<BaseTensorPtr> GetNewTensor(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                        const std::vector<BaseTensorPtr> &tensors) {
  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;
  std::vector<BaseTensorPtr> result{};
  auto input_shape = input_tensor->shape();
  if (input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension "
                             << input_shape.size();
  }
  if (tensors.size() > input_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', too many indices for tensor of dimension " << input_shape.size()
                             << " (got " << tensors.size() << ")";
  }
  bool needCast = false;
  TypeId indicesDtype = tensors[0]->data_type();
  for (const auto &tensor : tensors) {
    auto type_id = tensor->data_type();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError)
        << "For 'InplaceIndexPut', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      auto shape = tensor->shape();
      auto rank = SizeToLong(shape.size());
      for (int64_t j = 0; j < rank; j++) {
        auto srcIdx = result.size() + j;
        if (shape[j] != input_shape[srcIdx]) {
          MS_EXCEPTION(ValueError) << "For 'InplaceIndexPut', the shape of the mask " << tensor->ElementsNum()
                                   << " at index " << j << " does not match the shape of the indexed tensor "
                                   << input_shape << " at index " << srcIdx;
        }
      }
      auto nonzero_op = CREATE_PYBOOST_OP(InnerNonZero, device_name);
      auto nonzero_tensor = nonzero_op->Call(tensor);
      for (int64_t j = 0; j < rank; j++) {
        const auto dim = std::make_shared<Int64Imm>(kIndex0);
        const auto index = std::make_shared<Int64Imm>(j);
        auto select_op = CREATE_PYBOOST_OP(SelectExt, device_name);
        auto select_tensor = select_op->Call(nonzero_tensor, dim, index);
        result.emplace_back(select_tensor);
      }
    } else {
      result.emplace_back(tensor);
    }
    if (indicesDtype != type_id) {
      needCast = true;
    }
  }
  if (needCast) {
    for (size_t i = 0; i < result.size(); i++) {
      if (result[i]->data_type() == kNumberTypeInt32) {
        result[i] = PyBoostUtils::CastTensor(result[i], kNumberTypeInt64, device_name);
      }
    }
  }
  return result;
}
}  // namespace

std::tuple<ShapeVector, ShapeArray> TransposeToFront(const ShapeVector &x_shape, const ShapeArray &index_shapes) {
  ShapeVector fin_x_shape;
  ShapeArray fin_index_shapes;
  for (size_t i = 0; i < index_shapes.size(); ++i) {
    auto idx_shape = index_shapes[i];
    auto x_size = x_shape[i];
    if (!(idx_shape.size() == 9 && std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; }))) {
      fin_index_shapes.push_back(idx_shape);
      fin_x_shape.push_back(x_size);
    }
  }

  for (size_t i = 0; i < index_shapes.size(); ++i) {
    auto idx_shape = index_shapes[i];
    auto x_size = x_shape[i];
    if (idx_shape.size() == 9 && std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; })) {
      fin_index_shapes.push_back(idx_shape);
      fin_x_shape.push_back(x_size);
    }
  }
  return std::make_tuple(std::move(fin_x_shape), std::move(fin_index_shapes));
}

bool IndexContiguous(const ShapeArray &index_shape) {
  auto isEmpty = [](const ShapeVector &idx_shape) {
    return idx_shape.size() == 9 && std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; });
  };
  auto isNoEmpty = [](const ShapeVector &idx_shape) {
    return !(idx_shape.size() == 9 && std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; }));
  };
  auto start = std::find_if(index_shape.begin(), index_shape.end(), isNoEmpty);
  auto stop = std::find_if(index_shape.rbegin(), index_shape.rend(), isNoEmpty);
  auto it = std::find_if(start, stop.base(), isEmpty);
  return it == stop.base();
}

ShapeVector ExpandShape(const ShapeVector &a, const ShapeVector &b) {
  // infer a complete shape
  int64_t dimsA = a.size();
  int64_t dimsB = b.size();
  int64_t ndim = dimsA > dimsB ? dimsA : dimsB;
  ShapeVector expanded_shape(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    int64_t offset = ndim - 1 - i;
    int64_t dimA = dimsA - 1 - offset;
    int64_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;
    if (sizeA != sizeB && sizeA != 1 && sizeB != 1) {
      MS_EXCEPTION(ValueError) << "For 'Index'"
                               << ", the size of tensor 'a' should match the size of tensor 'b' at dimension " << i
                               << ", but got a size " << sizeA << ", b size " << sizeB << ".";
    }
    expanded_shape[i] = sizeA == 1 ? sizeB : sizeA;
  }
  return expanded_shape;
}

ShapeArray ExpandIndexShape(const ShapeArray &to_expand) {
  // expands a list of Tensors; ignores empty tensors
  bool first = true;
  ShapeVector tmp_shape;
  ShapeArray expanded_shapes(to_expand.size());
  for (size_t i = 0; i < to_expand.size(); ++i) {
    auto elem_shape = to_expand[i];
    if (elem_shape.size() == 9 && std::all_of(elem_shape.begin(), elem_shape.end(), [](int i) { return i == 0; })) {
      expanded_shapes[i] = elem_shape;
      continue;
    }
    if (first) {
      tmp_shape = elem_shape;
      first = false;
    } else {
      tmp_shape = ExpandShape(tmp_shape, elem_shape);
    }
    expanded_shapes[i] = tmp_shape;
  }
  return expanded_shapes;
}

ShapeVector GetIndexShape(ShapeVector x_shape, ShapeArray indices) {
  // 1. Get the expanded index shape.
  auto expand_index_shapes = ExpandIndexShape(indices);
  while (expand_index_shapes.size() < x_shape.size()) {
    ShapeVector empty_shape = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    expand_index_shapes.emplace_back(empty_shape);
  }

  // 2. If the non-empty tensor in the index is not contiguous, move it to the front to make it contiguous.
  ShapeVector fin_x_shape = x_shape;
  ShapeArray fin_index_shapes = expand_index_shapes;
  if (!IndexContiguous(expand_index_shapes)) {
    fin_x_shape.clear();
    fin_index_shapes.clear();
    std::tie(fin_x_shape, fin_index_shapes) = TransposeToFront(x_shape, expand_index_shapes);
  }

  // 3. Get the position and shape information of non-empty tensors in an index.
  int64_t dims_before = 0;
  int64_t dims_after = 0;
  int64_t dims_indexed = 0;
  ShapeVector replacement_shape;
  ShapeVector indexed_sizes;
  for (size_t dim = 0; dim < fin_index_shapes.size(); dim++) {
    auto idx_shape = fin_index_shapes[dim];
    if (idx_shape.size() == 9 && std::all_of(idx_shape.begin(), idx_shape.end(), [](int i) { return i == 0; })) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = idx_shape;
      indexed_sizes.push_back(fin_x_shape[dim]);
    }
  }

  // 4. If the input tensor has shape 0 but the index tensor does not, report error.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    MS_EXCEPTION(ValueError) << "For 'Index', if the input tensor of dimension with size 0"
                             << ", the index tensor should same"
                             << ", but index is out of bounds for dimension with size 0";
  }

  // 5. Replaces the indexed part in the shape of the input tensor with the shape of the index.
  ShapeVector out_shape(fin_x_shape);
  int64_t end = dims_before + dims_indexed;
  out_shape.erase(out_shape.begin() + dims_before, out_shape.begin() + end);
  out_shape.insert(out_shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());

  return out_shape;
}

bool IsExpandableTo(ShapeVector shape, ShapeVector desired) {
  // True if `shape` can be broadcasted to `desired`
  size_t ndim = shape.size();
  size_t target_dim = desired.size();
  if (ndim > target_dim) {
    return false;
  }
  for (size_t i = 0; i < ndim; i++) {
    int64_t size = shape[ndim - i - 1];
    int64_t target = desired[target_dim - i - 1];
    if (size != target && size != 1) {
      return false;
    }
  }
  return true;
}

tensor::BaseTensorPtr InplaceIndexPutAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                     const BaseTensorPtr &input_tensor,
                                                     const ValueTuplePtr &indices_tensor_list,
                                                     const BaseTensorPtr &values_tensor, const BoolImmPtr &accumulate) {
  op->set_outputs({input_tensor});

  const auto &input_shape = input_tensor->shape();
  const auto &values_shape = values_tensor->shape();
  std::vector<BaseTensorPtr> indices_tensor_vector = ConvertValueTupleToVector<BaseTensorPtr>(indices_tensor_list);
  auto input_numel =
    std::accumulate(input_shape.begin(), input_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto values_numel =
    std::accumulate(values_shape.begin(), values_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (input_numel == 0 || values_numel == 0 || indices_tensor_vector.size() == 0) {
    return op->output(0);
  }

  auto new_indices_tensor_vector = GetNewTensor(op, input_tensor, indices_tensor_vector);
  ShapeArray new_indices_shape;

  std::transform(new_indices_tensor_vector.begin(), new_indices_tensor_vector.end(),
                 std::back_insert_iterator(new_indices_shape),
                 [](const BaseTensorPtr tensor_ptr) { return tensor_ptr->shape(); });

  auto index_res_shape = GetIndexShape(input_shape, new_indices_shape);
  MS_CHECK_VALUE(
    IsExpandableTo(values_shape, index_res_shape),
    "For 'InplaceIndexPut', shape mismatch: value tensor of shape cannot be broadcast to indexing result of shape.");
  auto accumulate_imm = GetValue<bool>(accumulate);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, new_indices_tensor_vector,
                                values_tensor);

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_tensor, new_indices_tensor_vector, values_tensor, accumulate_imm]() {
      auto device_context = op->device_context();
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, new_indices_tensor_vector, values_tensor);

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnIndexPutImpl, device_context, op->stream_id(), input_tensor, new_indices_tensor_vector,
                   values_tensor, accumulate_imm, false);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
