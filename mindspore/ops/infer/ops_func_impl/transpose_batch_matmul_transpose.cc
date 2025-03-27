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

#include "infer/ops_func_impl/transpose_batch_matmul_transpose.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <set>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kTBMMTInputX1 = 0;
constexpr size_t kTBMMTInputX2 = 1;
constexpr size_t kTBMMTInputPermIn = 2;
constexpr size_t kTBMMTInputPermOut = 3;
constexpr size_t kTBMMTInputTransposeA = 4;
constexpr size_t kTBMMTInputTransposeB = 5;
constexpr size_t kTBMMTInputNum = 6;

size_t NormalizePermIdx(int64_t idx, size_t rank) {
  auto new_idx = idx >= 0 ? idx : (idx + SizeToLong(rank));
  return LongToSize(new_idx);
}
}  // namespace

void TransposeBatchMatmulTransposeMakeShape(ShapeVector *output, const ShapeVector x_shape, const ShapeVector w_shape,
                                            const ArrayValue<int64_t> &perm_in, const ArrayValue<int64_t> &perm_out,
                                            bool transpose_a, bool transpose_b) {
  size_t offset = kDim2;
  if (x_shape.empty() || w_shape.empty()) {
    return;
  }

  // transpose
  ShapeVector transpose_x_shape;
  for (size_t i = 0; i < perm_in.size(); ++i) {
    if (MS_UNLIKELY(perm_in.IsValueUnknown(i))) {
      transpose_x_shape.push_back(abstract::TensorShape::kShapeDimAny);
      continue;
    }

    auto dim = NormalizePermIdx(perm_in[i], x_shape.size());
    transpose_x_shape.push_back(x_shape[dim]);
  }

  // bmm
  ShapeVector long_input = transpose_x_shape.size() > w_shape.size() ? transpose_x_shape : w_shape;
  ShapeVector short_input = transpose_x_shape.size() > w_shape.size() ? w_shape : transpose_x_shape;
  size_t size_diff = long_input.size() - short_input.size();

  ShapeVector bmm_shape;
  bmm_shape.reserve(long_input.size());

  for (size_t i = 0; i < long_input.size() - offset; i++) {
    if (long_input[i] < 0) {
      bmm_shape.push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      bmm_shape.push_back(long_input[i] > short_input[i - size_diff] ? long_input[i] : short_input[i - size_diff]);
    } else {
      bmm_shape.push_back(long_input[i]);
    }
  }
  size_t x_offset = transpose_x_shape.size() - offset;
  size_t y_offset = w_shape.size() - offset;

  bmm_shape.push_back(transpose_x_shape[x_offset + (transpose_a ? 1 : 0)]);
  bmm_shape.push_back(w_shape[y_offset + (transpose_b ? 0 : 1)]);

  // transpose
  for (size_t i = 0; i < perm_out.size(); ++i) {
    if (MS_UNLIKELY(perm_out.IsValueUnknown(i))) {
      output->push_back(abstract::TensorShape::kShapeDimAny);
      continue;
    }

    auto dim = NormalizePermIdx(perm_out[i], long_input.size());
    output->push_back(bmm_shape[dim]);
  }
  return;
}

void CheckBMMInputSize(const std::string &op_name, const std::string &input_name, const ShapeVector &shape) {
  constexpr size_t minimum_dim_limit = 2;
  constexpr size_t maximum_dim_limit = 4;
  if (shape.size() < minimum_dim_limit || shape.size() > maximum_dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D/3D/4D Tensor, but got " << shape.size() << "D shape " << shape;
  }
}

void CheckTransposeInputAndOutputPerm(const std::string &op_name, const std::string &input_name_in,
                                      const ArrayValue<int64_t> &perm_in, const std::string &input_name_out,
                                      const ArrayValue<int64_t> &perm_out) {
  if (MS_UNLIKELY(perm_in.size() != perm_out.size())) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', size of the input '" << input_name_in
                             << "' should equal to the input '" << input_name_out << "' but got " << perm_in.size()
                             << " and " << perm_out.size();
  }

  for (size_t i = 0; i < perm_in.size(); ++i) {
    if (MS_UNLIKELY(perm_in.IsValueUnknown(i) || perm_out.IsValueUnknown(i))) {
      continue;
    }

    if (perm_in[i] != perm_out[i]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << i << "th value of the input '" << input_name_in
                               << "' should equal to the input '" << input_name_out << "' but got " << perm_in[i]
                               << " and " << perm_out[i];
    }
  }
}

void CheckTransposePerm(const std::string &op_name, const std::string &input_name, const std::size_t x_rank,
                        const ArrayValue<int64_t> &perm) {
  if (MS_UNLIKELY(perm.size() != x_rank)) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', size of the input '" << input_name
                             << "' should equal to rank of x, but got " << perm.size() << " and " << x_rank;
  }

  std::set<size_t> seen;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (MS_UNLIKELY(perm.IsValueUnknown(i))) {
      continue;
    }

    int64_t x_rank_long = SizeToLong(x_rank);
    if (MS_UNLIKELY(perm[i] < -x_rank_long || perm[i] >= x_rank_long)) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', size of the input '" << input_name
                               << "' must be in the range [" << -x_rank_long << ", " << x_rank_long << "), but got "
                               << perm[i];
    }

    auto dim = NormalizePermIdx(perm[i], x_rank);
    if (MS_UNLIKELY(seen.count(dim) != 0)) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                               << " should all be unique dim, but  " << dim << "  is not unique!";
    }
    seen.insert(dim);
  }
}

abstract::BaseShapePtr TransposeBatchMatmulTransposeFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, kTBMMTInputNum,
                                           primitive->name());
  const auto &prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kTBMMTInputX1]->GetShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a Tensor type, but got:" << input_args[kTBMMTInputX1]->ToString();
  }

  auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kTBMMTInputX2]->GetShape());
  if (w_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'w' must be a Tensor type, but got:" << input_args[kTBMMTInputX2]->ToString();
  }

  auto perm_in_res = GetArrayValue<int64_t>(input_args[kTBMMTInputPermIn]);
  auto perm_out_res = GetArrayValue<int64_t>(input_args[kTBMMTInputPermOut]);
  auto x_shp = x_shape_map[kShape];
  auto x_rank = x_shp.size();
  if (MS_UNLIKELY(!perm_in_res.has_value() || !perm_out_res.has_value())) {
    ShapeVector out_shape(x_rank, abstract::TensorShape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  auto transpose_a_op = GetScalarValue<bool>(input_args[kTBMMTInputTransposeA]->GetValue());
  auto transpose_b_op = GetScalarValue<bool>(input_args[kTBMMTInputTransposeB]->GetValue());
  if (MS_UNLIKELY(!transpose_a_op.has_value() || !transpose_b_op.has_value())) {
    ShapeVector out_shape(x_rank, abstract::TensorShape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  auto w_shp = w_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(w_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto transpose_a = transpose_a_op.value();
  auto transpose_b = transpose_b_op.value();
  auto perm_in = perm_in_res.value();
  auto perm_out = perm_out_res.value();
  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(w_shp);
  if (!dynamic_shape) {
    CheckBMMInputSize(prim_name, "x", x_shp);
    CheckBMMInputSize(prim_name, "w", w_shp);
    CheckTransposePerm(prim_name, "perm_in", x_rank, perm_in);
    CheckTransposePerm(prim_name, "perm_out", x_rank, perm_out);
    CheckTransposeInputAndOutputPerm(prim_name, "perm_in", perm_in, "perm_out", perm_out);
  }

  ShapeVector ret_shape;
  TransposeBatchMatmulTransposeMakeShape(&ret_shape, x_shp, w_shp, perm_in, perm_out, transpose_a, transpose_b);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr TransposeBatchMatmulTransposeFuncImpl::InferType(const PrimitivePtr &prim,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kTBMMTInputX1]->GetType());
  (void)types.emplace("w", input_args[kTBMMTInputX2]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kTBMMTInputX1]->GetType();
}
}  // namespace ops
}  // namespace mindspore
