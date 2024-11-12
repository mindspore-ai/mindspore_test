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
#include <memory>
#include <map>
#include <set>
#include <string>
#include <utility>
#include "infer/ops_func_impl/index.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

TypePtr IndexFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_LOG(EXCEPTION) << "Currently, the 'Index' supports only the pynative mode.";
  return input_args[kIndex0]->GetType();
}

TypePtrList IndexFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  return {input_type};
}

ShapeArray IndexFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto x_shape = x_tensor->shape();

  auto indices = input_values[kIndex1]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(indices);

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
  return {out_shape};
}

std::tuple<ShapeVector, ShapeArray> IndexFuncImpl::TransposeToFront(const ShapeVector &x_shape,
                                                                    const ShapeArray &index_shapes) const {
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

bool IndexFuncImpl::IndexContiguous(const ShapeArray &index_shape) const {
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

ShapeArray IndexFuncImpl::ExpandIndexShape(const ValueTuplePtr &to_expand) const {
  // expands a list of Tensors; ignores empty tensors
  bool first = true;
  ShapeVector tmp_shape;
  ShapeArray expanded_shapes(to_expand->size());
  for (size_t i = 0; i < to_expand->size(); ++i) {
    auto elem = to_expand->value()[i]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(elem);
    auto elem_shape = elem->shape();
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

ShapeVector IndexFuncImpl::ExpandShape(const ShapeVector &a, const ShapeVector &b) const {
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
REGISTER_SIMPLE_INFER(kNameIndex, IndexFuncImpl)
}  // namespace ops
}  // namespace mindspore
