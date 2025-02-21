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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/cosine_embedding_loss.h"
#include <string>
#include <unordered_set>
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore::prim {
void CheckCosineEmbeddingLossInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const std::unordered_set<TypeId> valid_types = {kNumberTypeBool,    kNumberTypeInt8,    kNumberTypeInt16,
                                                  kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,
                                                  kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  auto CheckType = [&](const AbstractBasePtr &input_tensor, const std::string &tensor_name) {
    auto type_id = input_tensor->GetType()->cast_ptr<TensorType>()->element()->type_id();
    if (valid_types.find(type_id) == valid_types.end()) {
      MS_EXCEPTION(TypeError)
        << "For 'CosineEmbeddingLoss', the type of '" << tensor_name
        << "' must be Tensor[Bool, Int8, Int16, Int32, Int64, UInt8, Float16, Float32, Float64], but got "
        << input_tensor->GetType();
    }
  };
  CheckType(input_args[kIndex0], "input1");
  CheckType(input_args[kIndex1], "input2");
  CheckType(input_args[kIndex2], "target");

  const auto &input1_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  const auto &input2_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  const auto &target_shape = input_args[kIndex2]->GetShape()->GetShapeVector();

  auto reduction_opt = GetScalarValue<int64_t>(input_args[kIndex4]->GetValue());
  if (reduction_opt.has_value()) {
    auto reduction_imm = reduction_opt.value();
    if (reduction_imm > static_cast<int64_t>(Reduction::NONE) ||
        reduction_imm < static_cast<int64_t>(Reduction::REDUCTION_SUM)) {
      MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, the value of reduction is invalid.";
    }
  }

  if (IsDynamicRank(input1_shape) || IsDynamicRank(input2_shape) || IsDynamicRank(target_shape)) {
    return;
  }

  auto input1_tensor_dim = SizeToLong(input1_shape.size());
  auto input2_tensor_dim = SizeToLong(input2_shape.size());
  auto target_tensor_dim = SizeToLong(target_shape.size());

  if (input1_tensor_dim != input2_tensor_dim) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, input1.shape and input2.shape should be equal.";
  }

  for (auto i = 0; i < input1_tensor_dim; ++i) {
    if (input1_shape[i] != input2_shape[i] && input1_shape[i] != abstract::Shape::kShapeDimAny &&
        input2_shape[i] != abstract::Shape::kShapeDimAny) {
      MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, input1.shape and input2.shape should be equal.";
    }
  }

  if (!(target_tensor_dim == kDim1 || target_tensor_dim == 0)) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, 0D or 1D target tensor expected, multi-target not supported.";
  }
  auto expect_input_tensor_dim = target_tensor_dim + 1;
  if (input1_tensor_dim != expect_input_tensor_dim || input2_tensor_dim != expect_input_tensor_dim) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, " << target_tensor_dim << "D target tensor expects "
                            << expect_input_tensor_dim << "D input tensors, but found inputs with " << input1_tensor_dim
                            << " and " << input2_tensor_dim << ".";
  }
}

BeginFunction(CosineEmbeddingLoss, input1_tensor, input2_tensor, target_tensor, margin, reduction) {
  constexpr float EPSILON = 1e-12;

  auto dim_tuple_ptr = Tuple(Rank(target_tensor));
  auto prod_sum =
    Call(Prim(SumExt), Call(Prim(Mul), input1_tensor, input2_tensor), dim_tuple_ptr, Value(false), Value(kNone));
  auto mag_square1 =
    Call(Prim(AddScalar),
         Call(Prim(SumExt), Call(Prim(Mul), input1_tensor, input1_tensor), dim_tuple_ptr, Value(false), Value(kNone)),
         Value(EPSILON), Value(1));
  auto mag_square2 =
    Call(Prim(AddScalar),
         Call(Prim(SumExt), Call(Prim(Mul), input2_tensor, input2_tensor), dim_tuple_ptr, Value(false), Value(kNone)),
         Value(EPSILON), Value(1));
  auto denom = Call(Prim(Sqrt), Call(Prim(Mul), mag_square1, mag_square2));
  auto cos = Call(Prim(Div), prod_sum, denom);

  auto zeros = Call(Prim(ZerosLikeExt), cos, Value(kNone));
  auto pos = Call(Prim(SubExt), Call(Prim(FillScalar), Shape(cos), Value(1), Value(kNone)), cos, Value(1));
  auto neg = Call(Prim(ClampMin), Call(Prim(SubScalar), cos, margin, Value(1)), Value(0));
  auto output_pos = Call(Prim(Select), Call(Prim(EqScalar), target_tensor, Value(1)), pos, zeros);
  auto output_neg = Call(Prim(Select), Call(Prim(EqScalar), target_tensor, Value(-1)), neg, zeros);
  auto output = Call(Prim(AddExt), output_pos, output_neg, Value(1));

  auto condition_none = Call(Prim(Equal), reduction, Value(static_cast<int64_t>(Reduction::NONE)));
  auto none_true_branch = [&]() { Return(output); };
  auto none_false_branch = [&]() {
    auto condition_mean = Call(Prim(Equal), reduction, Value(static_cast<int64_t>(Reduction::MEAN)));
    auto mean_true_branch = [&]() { Return(Call(Prim(MeanExt), output, Value(kNone), Value(false), Value(kNone))); };
    auto mean_false_branch = [&]() { Return(Call(Prim(SumExt), output, Value(kNone), Value(false), Value(kNone))); };
    Return(If(condition_mean, mean_true_branch, mean_false_branch, (output)));
  };
  Return(If(condition_none, none_true_branch, none_false_branch, (reduction, output)));
}
EndFunction(CosineEmbeddingLoss)
}  // namespace mindspore::prim
