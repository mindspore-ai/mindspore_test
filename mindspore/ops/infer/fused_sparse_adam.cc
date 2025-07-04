/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "infer/fused_sparse_adam.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace ops {
namespace fused_sparse_adam {
// "var","m","v","beta1_power","beta2_power","lr","beta1","beta2","epsilon","grad","indices"
constexpr size_t kVarIndex = 0;
constexpr size_t kMIndex = 1;
constexpr size_t kVIndex = 2;
constexpr size_t kBeta1PowerIndex = 3;
constexpr size_t kBeta2Powerndex = 4;
constexpr size_t kLrIndex = 5;
constexpr size_t kBeta1Index = 6;
constexpr size_t kBeta2Index = 7;
constexpr size_t kEpsilonIndex = 8;
constexpr size_t kGradIndex = 9;
constexpr size_t kIndicesIndex = 10;
constexpr size_t kFusedSparseAdamInputsNum = 11;

abstract::TupleShapePtr FusedSparseAdamInferShapeCommon(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args,
                                                        const abstract::BaseShapePtr &var_shape_r,
                                                        const abstract::BaseShapePtr &m_shape_r,
                                                        const abstract::BaseShapePtr &v_shape_r) {
  auto prim_name = primitive->name();
  auto outputs =
    std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>({var_shape_r, m_shape_r, v_shape_r}));
  for (auto &input : input_args) {
    if (input->GetShape()->IsDynamic()) {
      return outputs;
    }
  }
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kVarIndex]->GetShape())[kShape];
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kMIndex]->GetShape())[kShape];
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kVIndex]->GetShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndicesIndex]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kGradIndex]->GetShape())[kShape];

  (void)CheckAndConvertUtils::CheckValue("var_shape", var_shape, kEqual, "m_shape", m_shape, prim_name);
  (void)CheckAndConvertUtils::CheckValue("var_shape", var_shape, kEqual, "v_shape", v_shape, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("indices rank", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("grad rank", SizeToLong(grad_shape.size()), kGreaterEqual, 1, prim_name);
  (void)CheckAndConvertUtils::CheckValue("grad_shape[0]", grad_shape[0], kEqual, "indices_shape[0]", indices_shape[0],
                                         prim_name);
  // grad_shape[1:] == var_shape[1:] while grad_shape[0] == indices_shape[0]
  if (var_shape.size() > 1) {
    auto expect_shape = indices_shape;
    (void)std::copy(var_shape.begin() + 1, var_shape.end(), std::back_inserter(expect_shape));
    if (grad_shape != expect_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the shape of updates must be [] or "
                               << "grad_shape = indices_shape + var_shape[1:], but got var_shape: " << var_shape
                               << ", indices_shape: " << indices_shape << ", grad_shape: " << grad_shape << ".";
    }
  }
  return outputs;
}

abstract::TupleShapePtr FusedSparseAdamInferShapeIner(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  // "var","m","v","beta1_power","beta2_power","lr","beta1","beta2","epsilon","grad","indices"

  // the output is useless, so we don't have to focus on the output shape, cannot return 1
  auto var_shape_r = input_args[kVarIndex]->Broaden()->GetShape();
  auto m_shape_r = input_args[kMIndex]->Broaden()->GetShape();
  auto v_shape_r = input_args[kVIndex]->Broaden()->GetShape();
  return FusedSparseAdamInferShapeCommon(primitive, input_args, var_shape_r, m_shape_r, v_shape_r);
}

abstract::TupleShapePtr FusedSparseAdamInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  // "var","m","v","beta1_power","beta2_power","lr","beta1","beta2","epsilon","grad","indices"

  // the output is useless, so we don't have to focus on the output shape, cannot return 1
  auto var_shape_r = input_args[kVarIndex]->GetShape();
  auto m_shape_r = input_args[kMIndex]->GetShape();
  auto v_shape_r = input_args[kVIndex]->GetShape();
  return FusedSparseAdamInferShapeCommon(primitive, input_args, var_shape_r, m_shape_r, v_shape_r);
}

TypePtr FusedSparseAdamInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  // "var","m","v","beta1_power","beta2_power","lr","beta1","beta2","epsilon","grad","indices"
  auto prim_name = prim->name();
  std::map<std::string, TypePtr> types = {{"var", input_args[kVarIndex]->GetType()},
                                          {"m", input_args[kMIndex]->GetType()},
                                          {"v", input_args[kVIndex]->GetType()},
                                          {"grad", input_args[kGradIndex]->GetType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types_with_complex, prim_name);

  types = {
    {"beta1_power", input_args[kBeta1PowerIndex]->GetType()},
    {"beta2_power", input_args[kBeta2Powerndex]->GetType()},
    {"lr", input_args[kLrIndex]->GetType()},
    {"beta1", input_args[kBeta1Index]->GetType()},
    {"beta2", input_args[kBeta2Index]->GetType()},
    {"epsilon", input_args[kEpsilonIndex]->GetType()},
  };
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types, {kFloat16, kFloat32}, prim_name);

  auto indices_dtype = input_args[kIndicesIndex]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_dtype, {kInt32}, prim_name);

  auto type = input_args[kVarIndex]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type});
}
}  // namespace fused_sparse_adam
void FusedSparseAdam::set_use_locking(bool use_locking) {
  (void)this->AddAttr(kUseLocking, api::MakeValue(use_locking));
}

bool FusedSparseAdam::get_use_locking() const {
  auto value_ptr = GetAttr(kUseLocking);
  return GetValue<bool>(value_ptr);
}

void FusedSparseAdam::set_use_nesterov(bool use_nesterov) {
  (void)this->AddAttr(kUseNesterov, api::MakeValue(use_nesterov));
}

bool FusedSparseAdam::get_use_nesterov() const {
  auto value_ptr = GetAttr(kUseNesterov);
  return GetValue<bool>(value_ptr);
}

void FusedSparseAdam::Init(bool use_locking, bool use_nesterov) {
  this->set_use_locking(use_locking);
  this->set_use_nesterov(use_nesterov);
}

MIND_API_OPERATOR_IMPL(FusedSparseAdam, BaseOperator);
AbstractBasePtr FusedSparseAdamInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(fused_sparse_adam::kFusedSparseAdamInputsNum), op_name);
  auto types = fused_sparse_adam::FusedSparseAdamInferType(primitive, input_args);
  auto shapes = fused_sparse_adam::FusedSparseAdamInferShapeIner(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class OPS_API AGFusedSparseAdamInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return fused_sparse_adam::FusedSparseAdamInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return fused_sparse_adam::FusedSparseAdamInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FusedSparseAdamInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FusedSparseAdam, prim::kPrimFusedSparseAdam, AGFusedSparseAdamInfer, false);
}  // namespace ops
}  // namespace mindspore
