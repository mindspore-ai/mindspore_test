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

#include "infer/ops_func_impl/batch_mat_mul.h"
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include "mindspore/ops/op_def/op_name.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {

void BatchMatMulMakeShape(ShapeVector *output, const ShapeVector xshp, const ShapeVector yshp, bool transpose_a,
                          bool transpose_b) {
  size_t offset = kDim2;
  if (xshp.empty() || yshp.empty()) {
    return;
  }
  ShapeVector long_input = xshp.size() > yshp.size() ? xshp : yshp;
  ShapeVector short_input = xshp.size() > yshp.size() ? yshp : xshp;
  size_t size_diff = long_input.size() - short_input.size();
  for (size_t i = 0; i < long_input.size() - offset; i++) {
    if (long_input[i] < 0) {
      output->push_back(abstract::Shape::kShapeDimAny);
    } else if (i >= size_diff) {
      output->push_back(long_input[i] > short_input[i - size_diff] ? long_input[i] : short_input[i - size_diff]);
    } else {
      output->push_back(long_input[i]);
    }
  }
  size_t x_offset = xshp.size() - offset;
  size_t y_offset = yshp.size() - offset;
  output->push_back(xshp[x_offset + (transpose_a ? 1 : 0)]);
  output->push_back(yshp[y_offset + (transpose_b ? 0 : 1)]);
  return;
}

void CheckBatchMatmulInputSize(const std::string &op_name, const std::string &input_name, const ShapeVector &shape) {
  constexpr size_t dim_limit = 2;
  if (shape.size() < dim_limit) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', the input '" << input_name
                             << "' must be a 2D or higher dimensional Tensor, but got " << shape.size() << "D shape "
                             << shape;
  }
}

void CheckBatchMatmulInputWhetherCanBeMul(const std::string &name, const ShapeVector &x_shape,
                                          const ShapeVector &y_shape, bool transpose_a, bool transpose_b) {
  ShapeVector x_mat_shape(x_shape.end() - SizeToLong(kDim2), x_shape.end());
  ShapeVector y_mat_shape(y_shape.end() - SizeToLong(kDim2), y_shape.end());
  int64_t x_col = x_mat_shape[static_cast<size_t>(!transpose_a)];
  int64_t y_row = y_mat_shape[static_cast<size_t>(transpose_b)];
  if (std::find(x_shape.begin(), x_shape.end(), -1) == x_shape.end() &&
      std::find(y_shape.begin(), y_shape.end(), -1) == y_shape.end()) {
    if (x_col != y_row) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", the row of the input 'y' should be same as the col of the input 'x', with x shape "
                               << x_shape << "(transpose_a=" << transpose_a << "), y shape " << y_shape
                               << "(transpose_b=" << transpose_b << ")";
    }
  }
}

void CheckBatchMatmulInputWhetherCanBeBroadcast(const std::string &name, const ShapeVector &x_shape,
                                                const ShapeVector &y_shape) {
  ShapeVector x_batch(x_shape.begin(), x_shape.end() - SizeToLong(kDim2));
  ShapeVector y_batch(y_shape.begin(), y_shape.end() - SizeToLong(kDim2));
  if (x_batch == y_batch) {
    return;
  }

  // todo: delete after broadcast shape is supported on cpu/gpu
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kGPUDevice || device_target == kCPUDevice) {
    if (x_batch.size() != y_batch.size()) {
      MS_EXCEPTION(ValueError) << "For " << name << ", inputs shape cannot be broadcast on CPU/GPU, with x shape "
                               << x_shape << ", y shape " << y_shape;
    }

    for (size_t i = 0; i < x_batch.size(); ++i) {
      if (x_batch[i] != y_batch[i]) {
        MS_EXCEPTION(ValueError) << "For " << name << ", inputs shape cannot be broadcast on CPU/GPU, with x shape "
                                 << x_shape << ", y shape " << y_shape;
      }
    }
  }

  size_t min_size = std::min(x_batch.size(), y_batch.size());
  for (int64_t i = 0; i < SizeToLong(min_size); ++i) {
    auto x = *(x_batch.rbegin() + i);
    auto y = *(y_batch.rbegin() + i);
    if (x != 1 && y != 1 && x != y) {
      MS_EXCEPTION(ValueError) << "For " << name
                               << ", one of the input's batch dim must be equal to another input's peer batch dim, or "
                                  "be equal to 1, or be empty, but got "
                               << x << " and " << y << ", with x shape " << x_shape << ", y shape " << y_shape;
    }
  }
}

BaseShapePtr BatchMatMulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto constexpr kBatchMatmulInputNum = 4;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, kBatchMatmulInputNum,
                                           primitive->name());
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape());
  auto y_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape());
  if (x_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a Tensor type, but got:" << input_args[0]->ToString();
  }
  if (y_shape_map.empty()) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'y' must be a Tensor type, but got:" << input_args[1]->ToString();
  }

  auto transpose_a_op = GetScalarValue<bool>(input_args[2]->GetValue());
  auto transpose_b_op = GetScalarValue<bool>(input_args[3]->GetValue());

  if (!transpose_a_op.has_value()) {
    return input_args[0]->GetShape()->Clone();
  }

  if (!transpose_b_op.has_value()) {
    return input_args[1]->GetShape()->Clone();
  }

  auto transpose_a = transpose_a_op.value();
  auto transpose_b = transpose_b_op.value();

  auto x_shp = x_shape_map[kShape];
  auto y_shp = y_shape_map[kShape];
  if (IsDynamicRank(x_shp) || IsDynamicRank(y_shp)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  bool dynamic_shape = IsDynamic(x_shp) || IsDynamic(y_shp);
  if (!dynamic_shape) {
    CheckBatchMatmulInputSize(prim_name, "x", x_shp);
    CheckBatchMatmulInputSize(prim_name, "y", y_shp);
    CheckBatchMatmulInputWhetherCanBeMul(prim_name, x_shp, y_shp, transpose_a, transpose_b);
    CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x_shp, y_shp);
  }

  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, y_shp, transpose_a, transpose_b);
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr BatchMatMulFuncImpl::InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,     kUInt32,
                                         kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->GetType());
  (void)types.emplace("w", input_args[1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  TypePtr x_type = input_args[0]->GetType();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if ((x_type->type_id() == TypeId::kNumberTypeInt8 || x_type->ToString() == "Tensor[Int8]") &&
      device_target == kAscendDevice) {
    x_type = std::make_shared<TensorType>(kInt32);
  }
  if (prim->HasAttr("cast_type")) {
    auto out_type = prim->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', MatMul cast_type must be a 'Type', but got: '"
                               << out_type << "'.";
    }
    auto out_type_ptr = out_type->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(out_type_ptr);
    x_type = std::make_shared<TensorType>(out_type_ptr);
  }
  return x_type;
}

TypePtrList BatchMatMulFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  TypePtr ret_type = x_tensor->Dtype();
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "BatchMatMul cast_type must be a `Type`";
    }
    ret_type = out_type->cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_type);
  }
  const auto x_type = x_tensor->Dtype();
  const auto y_type = y_tensor->Dtype();
  auto op_name = primitive->name();
  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (x_type->type_id() == TypeId::kNumberTypeInt8 && device_target == kAscendDevice) {
    ret_type = kInt32;
  }
  return {ret_type};
}

ShapeArray BatchMatMulFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  const auto &y_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);

  const auto &x_shp = x_tensor->shape();
  const auto &y_shp = y_tensor->shape();

  auto transpose_a_op = GetScalarValue<bool>(input_values[kInputIndex2]);
  auto transpose_b_op = GetScalarValue<bool>(input_values[kInputIndex3]);

  auto transpose_a = transpose_a_op.value();
  auto transpose_b = transpose_b_op.value();

  auto prim_name = primitive->name();
  CheckBatchMatmulInputSize(prim_name, "x", x_shp);
  CheckBatchMatmulInputSize(prim_name, "y", y_shp);
  CheckBatchMatmulInputWhetherCanBeMul(prim_name, x_shp, y_shp, transpose_a, transpose_b);
  CheckBatchMatmulInputWhetherCanBeBroadcast(prim_name, x_shp, y_shp);
  ShapeVector ret_shape;
  BatchMatMulMakeShape(&ret_shape, x_shp, y_shp, transpose_a, transpose_b);
  return {ret_shape};
}
REGISTER_SIMPLE_INFER(kNameBatchMatMul, BatchMatMulFuncImpl)
}  // namespace ops
}  // namespace mindspore
