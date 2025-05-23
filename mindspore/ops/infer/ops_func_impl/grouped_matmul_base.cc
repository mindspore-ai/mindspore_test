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
#include "infer/ops_func_impl/grouped_matmul_base.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>

#include "ops/ops_func_impl/op_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kMultiOutGroupType = -1;
}  // namespace
std::pair<ShapeArray, ShapeArray> GroupedMatmulBaseFuncImpl::FetchInputAndWeightShapes(
  const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  ShapeArray x_shapes;
  ShapeArray w_shapes;
  if (MS_LIKELY(input_infos[idxes_.x]->IsSequence())) {
    FetchGroupInfo(primitive, input_infos);
    auto FetchTupleTensorShapeFunc = [](const InferInfoPtr &tensors) {
      const auto &elements = tensors->GetSequenceElements();
      ShapeArray shapes;
      std::transform(elements.begin(), elements.end(), std::back_inserter(shapes),
                     [](const InferInfoPtr &info) { return info->GetShape(); });
      return shapes;
    };
    // get tuple_x_shape in compile phase
    x_shapes = FetchTupleTensorShapeFunc(input_infos[idxes_.x]);
    // get tuple_w_shape in compile phase
    w_shapes = FetchTupleTensorShapeFunc(input_infos[idxes_.weight]);
  } else {
    // Runtime phase: the element in input_args is KernelTensor. (tuple is expanded)
    auto tuple_len = GetValue<std::vector<int64_t>>(primitive->GetAttr("group_info"));
    size_t x_idx_end = LongToSize(tuple_len[0]);
    size_t w_idx_end = LongToSize(tuple_len[0] + tuple_len[1]);
    std::transform(input_infos.begin(), input_infos.begin() + x_idx_end, std::back_inserter(x_shapes),
                   [](const InferInfoPtr &info) { return info->GetShape(); });
    std::transform(input_infos.begin() + x_idx_end, input_infos.begin() + w_idx_end, std::back_inserter(w_shapes),
                   [](const InferInfoPtr &info) { return info->GetShape(); });
  }
  return std::make_pair(std::move(x_shapes), std::move(w_shapes));
}

void GroupedMatmulBaseFuncImpl::CheckInputAndWeightShapeForSingleOutput(const PrimitivePtr &primitive,
                                                                        const ShapeVector &x_shape,
                                                                        const ShapeVector &w_shape, int64_t group_type,
                                                                        bool transpose_b) const {
  const auto &op_name = primitive->name();
  static std::unordered_map<int64_t, std::pair<size_t, size_t>> expect_xw_ranks{
    {0, std::make_pair(2, 3)},  // group_type 0, split_item 3, x_rank = 2, w_rank = 3
    {2, std::make_pair(2, 2)}   // group_type 2, split_item 3, x_rank = 2, w_rank = 2
  };
  auto &[expect_x_rank, expect_w_rank] = expect_xw_ranks[group_type];

  if (x_shape.size() != expect_x_rank) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', when group_type is " << group_type
                             << " and split_item is 3, the x[0] must be " << expect_x_rank
                             << "D Tensor. But got x[0]'s shape: " << x_shape;
  }
  if (w_shape.size() != expect_w_rank) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', when group_type is " << group_type
                             << " and split_item is 3, the w[0] must be " << expect_w_rank
                             << "D Tensor. But got w[0]'s shape :" << w_shape;
  }
  auto x_k = x_shape.back();
  ShapeValueDType w_k = 0;
  if (transpose_b) {
    w_k = w_shape[w_shape.size() - kInputIndex1];
  } else {
    w_k = w_shape[w_shape.size() - kInputIndex2];
  }
  if (MS_UNLIKELY(x_k != abstract::Shape::kShapeDimAny && w_k != abstract::Shape::kShapeDimAny && x_k != w_k)) {
    auto expect_w_shape = group_type == 0 ? "(e, k, n)" : "(k, n)";
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', when group_type is " << group_type
                             << ", x[0]'s shape should be (m, k), w[0]'s shape should be " << expect_w_shape
                             << ", but got x[0]'s shape: " << x_shape << ", w[0]'s shape: " << w_shape;
  }
}

ShapeArray GroupedMatmulBaseFuncImpl::InferShapeForSingleOutput(const PrimitivePtr &primitive,
                                                                const ShapeArray &x_shapes, const ShapeArray &w_shapes,
                                                                int64_t group_list_size, int64_t group_type,
                                                                bool transpose_b, bool is_int4) const {
  if (MS_UNLIKELY(x_shapes.size() != kIndex1 || w_shapes.size() != kIndex1)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', when split_item is 3. the size of x and weight should both be 1, but got x's size "
                             << x_shapes.size() << ", and weight's size " << w_shapes.size();
  }

  const auto &x_shape = x_shapes[0];
  const auto &w_shape = w_shapes[0];
  auto is_x_dyn_rank = IsDynamicRank(x_shape);
  auto is_w_dyn_rank = IsDynamicRank(w_shape);
  if (!is_x_dyn_rank && !is_w_dyn_rank) {
    CheckInputAndWeightShapeForSingleOutput(primitive, x_shape, w_shape, group_type, transpose_b);
  }
  auto m = is_x_dyn_rank ? abstract::Shape::kShapeDimAny : x_shape[x_shape.size() - 2];
  auto n = abstract::Shape::kShapeDimAny;
  if (!is_w_dyn_rank) {
    n = transpose_b ? w_shape[w_shape.size() - kInputIndex2] : w_shape.back();
    if (is_int4) {
      n = n << 1;
    }
  }

  std::vector<int64_t> res_shape;
  if (group_type == 0) {
    // x.shape [m, k], w.shape [e, k, n], y.shape [m, n]
    res_shape = std::vector<int64_t>{m, n};
  } else {
    // x.shape [m, k], w.shape [k, n], y.shape [b, m, n]
    res_shape = std::vector<int64_t>{group_list_size, m, n};
  }
  return {std::move(res_shape)};
}

void GroupedMatmulBaseFuncImpl::CheckInputAndWeightShapeForMultiOutput(const PrimitivePtr &primitive,
                                                                       const ShapeVector &x_shape,
                                                                       const ShapeVector &w_shape, size_t i) const {
  const auto &op_name = primitive->name();
  if (MS_UNLIKELY(!IsDynamicRank(x_shape) && (x_shape.size() < kIndex2 || x_shape.size() > kIndex6))) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', when group_type is -1 and split_item is 0, the tensor in 'x' must be 2-6D, but got x[" << i << "]'s shape "
      << x_shape;
  }
  if (MS_UNLIKELY(!IsDynamicRank(w_shape) && w_shape.size() != kIndex2)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', when group_type is -1 and split_item is 0, the tensor in 'w' must be 2D, but got w["
                             << i << "]'s shape " << w_shape;
  }
  auto x_k = IsDynamicRank(x_shape) ? abstract::Shape::kShapeDimAny : x_shape.back();
  auto w_k = IsDynamicRank(w_shape) ? abstract::Shape::kShapeDimAny : w_shape.front();
  if (MS_UNLIKELY(x_k != abstract::Shape::kShapeDimAny && w_k != abstract::Shape::kShapeDimAny && x_k != w_k)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', when group_type is -1 and split_item is 0, the back in x["
                             << i << "]'s shape should be equal to the first in w[" << i << "]'s shape, but got x[" << i
                             << "]'s shape : " << x_shape << ", w[" << i << "]'s shape : " << w_shape;
  }
}

ShapeArray GroupedMatmulBaseFuncImpl::InferShapeForMultiOutput(const PrimitivePtr &primitive,
                                                               const ShapeArray &x_shapes,
                                                               const ShapeArray &w_shapes) const {
  if (MS_UNLIKELY(x_shapes.size() != w_shapes.size())) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', when group_type is -1 and split_item is 0, x's size should be equal to weight, but got ."
      << x_shapes.size() << " and " << w_shapes.size();
  }

  ShapeArray output_shapes;
  for (size_t i = 0; i < x_shapes.size(); i++) {
    const auto &x_shape = x_shapes[i];
    const auto &w_shape = w_shapes[i];
    CheckInputAndWeightShapeForMultiOutput(primitive, x_shape, w_shape, i);
    if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
      (void)output_shapes.emplace_back(ShapeVector{abstract::TensorShape::kShapeRankAny});
    } else {
      auto res_shape = x_shape;
      res_shape.back() = IsDynamicRank(w_shape) ? abstract::Shape::kShapeDimAny : w_shape.back();
      (void)output_shapes.emplace_back(std::move(res_shape));
    }
  }
  return output_shapes;
}

ShapeArray GroupedMatmulBaseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto [x_shapes, w_shapes] = FetchInputAndWeightShapes(primitive, input_infos);
  const auto input_num = SizeToLong(input_infos.size());
  const auto group_type_idx = input_num + idxes_.group_type_offset;
  auto group_type_opt = input_infos[group_type_idx]->GetScalarValue<int64_t>();
  MS_ASSERT(group_type_opt.has_value());
  auto group_type = group_type_opt.value();
  if (group_type == -1) {
    return InferShapeForMultiOutput(primitive, x_shapes, w_shapes);
  }

  auto group_list_size = FetchGroupListSize(primitive, input_infos);
  const auto transpose_b_idx = input_num + idxes_.transpose_b_offset;
  auto transpose_b = GetTransposeValue(input_infos, transpose_b_idx);
  bool is_int4 = false;
  if (MS_LIKELY(input_infos[idxes_.weight]->IsSequence())) {
    const auto &w_tensors = input_infos[idxes_.weight]->GetSequenceElements();
    MS_ASSERT(w_tensors.size() > 0);
    is_int4 = w_tensors[0]->GetType() == kNumberTypeInt4;
  } else {
    is_int4 = input_infos[idxes_.weight]->GetType() == kNumberTypeInt4;
  }

  return InferShapeForSingleOutput(primitive, x_shapes, w_shapes, group_list_size, group_type, transpose_b, is_int4);
}

std::pair<int32_t, int64_t> GroupedMatmulBaseFuncImpl::CommonCheckValidation(
  const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto input_num = SizeToLong(input_infos.size());
  const auto group_type_idx = input_num + idxes_.group_type_offset;
  auto group_type_opt = input_infos[group_type_idx]->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!group_type_opt.has_value())) {
    MS_EXCEPTION(RuntimeError) << "For '" << primitive->name() << "', group_type should not be dynamic.";
  }
  auto group_type = group_type_opt.value();
  static std::set<int64_t> valid_group_type_list{-1, 0, 2};
  if (MS_UNLIKELY(valid_group_type_list.find(group_type) == valid_group_type_list.end())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_type should be -1, 0 or 2, but got "
                             << group_type;
  }

  const auto split_item_idx = idxes_.split_item_offset + input_num;
  auto split_item_opt = input_infos[split_item_idx]->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!split_item_opt.has_value())) {
    return std::make_pair(OP_CHECK_RETRY, group_type);
  }
  static std::unordered_map<int64_t, int64_t> valid_split_item_map{{-1, 0}, {0, 3}, {2, 3}};
  int64_t expect_split_item = valid_split_item_map[group_type];
  auto split_item = split_item_opt.value();
  if (MS_UNLIKELY(split_item != expect_split_item)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', when group_type is " << group_type
                             << ", split_item should be " << expect_split_item << ", but got " << split_item;
  }

  return std::make_pair(OP_CHECK_SUCCESS, group_type);
}

int32_t GroupedMatmulBaseFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto [common_check_result, group_type] = CommonCheckValidation(primitive, input_infos);
  auto private_check_result = PrivateCheckValidation(primitive, input_infos, group_type);
  return common_check_result + private_check_result < 0 ? OP_CHECK_RETRY : OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
