/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/matmul_fusion_utils.h"
#include <vector>
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace ops {
BaseShapePtr MatmulFusionUtils::InferenceMultiMatmulInferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  constexpr size_t kSize2 = 2;
  constexpr size_t kSize3 = 3;
  const size_t x_rank = x_shape.size();
  const size_t w_rank = w_shape.size();
  MS_CHECK_VALUE((x_rank == kSize3 || x_rank == kSize2),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', x_rank should be 3 or 2."));

  MS_CHECK_VALUE(w_rank == kSize2,
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', w_rank should be 2."));

  auto b = x_rank == kSize3 ? x_shape[0] : 1;  // in matmul, m = b * s
  auto s = x_shape[x_rank - 2];
  auto k = x_shape[x_rank - 1];
  auto k0 = w_shape[1];
  MS_CHECK_VALUE(k == k0, CheckAndConvertUtils::FormatCommMsg(
                            "For '" + primitive->name() + "', the K axis of all inputs must have the same length."));

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', op must have attr 'n_lens'."));

  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));
  MS_CHECK_VALUE(
    (n_len_list.size() == kSize2 || n_len_list.size() == kSize3),
    CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', attr 'n_lens' must have 2 or 3 value."));

  ShapeVector output_0_shape = x_rank == kSize3 ? std::vector<ShapeValueDType>{b, s, n_len_list[kIndex0]}
                                                : std::vector<ShapeValueDType>{s, n_len_list[kIndex0]};
  ShapeVector output_1_shape = x_rank == kSize3 ? std::vector<ShapeValueDType>{b, s, n_len_list[kIndex1]}
                                                : std::vector<ShapeValueDType>{s, n_len_list[kIndex1]};
  std::vector<BaseShapePtr> shape_lists;
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_0_shape));
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_1_shape));
  if (n_len_list.size() == kSize3) {
    ShapeVector output_2_shape = x_rank == kSize3 ? std::vector<ShapeValueDType>{b, s, n_len_list[kIndex2]}
                                                  : std::vector<ShapeValueDType>{s, n_len_list[kIndex2]};
    (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_2_shape));
  }
  return std::make_shared<abstract::TupleShape>(shape_lists);
}

BaseShapePtr MatmulFusionUtils::InferenceMultiMatmulInferShapeFused(const PrimitivePtr &primitive,
                                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    MS_LOG(EXCEPTION) << "For " << op_name << ", dynamic rank is not supported";
  }
  constexpr size_t kSize2 = 2;
  constexpr size_t kSize3 = 3;
  const size_t x_rank = x_shape.size();
  const size_t w_rank = w_shape.size();
  MS_CHECK_VALUE(x_rank == kSize3,
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', x_rank should be 3."));

  MS_CHECK_VALUE(w_rank == kSize2,
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', w_rank should be 2."));

  auto b = x_shape[0];  // in matmul, m = b * s
  auto s = x_shape[1];
  auto k = x_shape[2];
  auto k0 = w_shape[1];
  MS_CHECK_VALUE(k == k0, CheckAndConvertUtils::FormatCommMsg(
                            "For '" + primitive->name() + "', the K axis of all inputs must have the same length."));

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', op must have attr 'n_lens'."));

  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));
  MS_CHECK_VALUE(
    (n_len_list.size() == kSize2 || n_len_list.size() == kSize3),
    CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', attr 'n_lens' must have 2 or 3 value."));

  ShapeVector output_0_shape = {b, s, n_len_list[kIndex0]};
  std::vector<BaseShapePtr> shape_lists;
  (void)shape_lists.emplace_back(std::make_shared<abstract::TensorShape>(output_0_shape));
  return std::make_shared<abstract::TupleShape>(shape_lists);
}

TuplePtr MatmulFusionUtils::InferenceMultiMatmulInferType(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->GetType();
  if (x_type == kInt8 && input_args[kInputIndex1]->GetType() == kInt8) {
    x_type = kFloat16;
  }

  MS_CHECK_VALUE(primitive->HasAttr("n_lens"),
                 CheckAndConvertUtils::FormatCommMsg("For '" + primitive->name() + "', op must have attr 'n_lens'."));
  std::vector<int64_t> n_len_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("n_lens"));
  constexpr size_t kSize2 = 2;
  constexpr size_t kSize3 = 3;
  if (n_len_list.size() == kSize3) {
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type, x_type});
  } else if (n_len_list.size() == kSize2) {
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, x_type});
  } else {
    MS_EXCEPTION(ValueError) << "n_lens's size must be 2 or 3 but got " << n_len_list.size();
  }
}

TypePtr MatmulFusionUtils::InferenceMultiMatmulInfer1OutputType(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->GetType();
  if (x_type == kInt8 && input_args[kInputIndex1]->GetType() == kInt8) {
    x_type = kFloat16;
  }
  return x_type;
}

TypePtr MatmulFusionUtils::FusedMatMulElemInferType(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args,
                                                    const int input_num) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  if (x_type->type_id() == TypeId::kNumberTypeInt8) {
    return kFloat16;
  }
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
