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

#include "kernel/ascend/pyboost/customize/cosine_embedding_loss.h"
#include <string>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "pyboost/functions/auto_generate/functions.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr CosineEmbeddingLossAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                         const BaseTensorPtr &input1_tensor,
                                                         const BaseTensorPtr &input2_tensor,
                                                         const BaseTensorPtr &target_tensor, const FP32ImmPtr &margin,
                                                         const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "CosineEmbeddingLoss Launch start";

  const std::unordered_set<TypeId> valid_types = {kNumberTypeBool,    kNumberTypeInt8,    kNumberTypeInt16,
                                                  kNumberTypeInt32,   kNumberTypeInt64,   kNumberTypeUInt8,
                                                  kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  auto CheckType = [&](const BaseTensorPtr &input_tensor, const std::string &tensor_name) {
    auto type_id = input_tensor->data_type();
    if (valid_types.find(type_id) == valid_types.end()) {
      MS_EXCEPTION(TypeError)
        << "For 'CosineEmbeddingLoss', the type of '" << tensor_name
        << "' must be Tensor[Bool, Int8, Int16, Int32, Int64, UInt8, Float16, Float32, Float64], but got "
        << input_tensor->Dtype();
    }
  };

  CheckType(input1_tensor, "input1");
  CheckType(input2_tensor, "input2");
  CheckType(target_tensor, "target");

  if (input1_tensor->shape() != input2_tensor->shape()) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, input1.shape and input2.shape should be equal.";
  }
  constexpr float EPSILON = 1e-12;
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  std::unordered_map<Reduction, int64_t> reduction_map = {
    {Reduction::REDUCTION_SUM, 2}, {Reduction::MEAN, 1}, {Reduction::NONE, 0}};
  auto iter = reduction_map.find(reduction_imm);
  if (iter == reduction_map.end()) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, the value of reduction is invalid.";
  }
  auto target_tensor_dim = SizeToLong(target_tensor->shape().size());
  if (!(target_tensor_dim == kDim1 || target_tensor_dim == 0)) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, 0D or 1D target tensor expected, multi-target not supported.";
  }
  auto input1_tensor_dim = SizeToLong(input1_tensor->shape().size());
  auto input2_tensor_dim = SizeToLong(input2_tensor->shape().size());
  auto expect_input_tensor_dim = target_tensor_dim + 1;
  if (input1_tensor_dim != expect_input_tensor_dim || input2_tensor_dim != expect_input_tensor_dim) {
    MS_EXCEPTION(TypeError) << "For CosineEmbeddingLoss, " << target_tensor_dim << "D target tensor expects "
                            << expect_input_tensor_dim << "D input tensors, but found inputs with " << input1_tensor_dim
                            << " and " << input2_tensor_dim << ".";
  }

  std::vector<ValuePtr> dim_ptr{MakeValue(static_cast<int64_t>(target_tensor_dim))};
  auto dim_tuple_ptr = std::make_shared<ValueTuple>(dim_ptr);
  auto prod_sum =
    sum_ext(mul(input1_tensor, input2_tensor), dim_tuple_ptr, std::make_shared<BoolImm>(False), std::nullopt);
  auto mag_square1 = add_scalar(
    sum_ext(mul(input1_tensor, input1_tensor), dim_tuple_ptr, std::make_shared<BoolImm>(False), std::nullopt),
    std::make_shared<FP64Imm>(EPSILON), std::make_shared<Int64Imm>(1));
  auto mag_square2 = add_scalar(
    sum_ext(mul(input2_tensor, input2_tensor), dim_tuple_ptr, std::make_shared<BoolImm>(False), std::nullopt),
    std::make_shared<FP64Imm>(EPSILON), std::make_shared<Int64Imm>(1));
  auto denom = sqrt(mul(mag_square1, mag_square2));
  auto cos = div(prod_sum, denom);
  auto zeros = zeros_like_ext(cos, std::nullopt);
  auto cos_shape = cos->shape();
  std::vector<ValuePtr> cos_shape_ptr;
  for (auto i = 0; i < SizeToLong(cos_shape.size()); i++) {
    cos_shape_ptr.emplace_back(std::make_shared<Int64Imm>(cos_shape[i]));
  }
  auto pos = sub_ext(fill_scalar(std::make_shared<ValueTuple>(cos_shape_ptr), std::make_shared<Int64Imm>(1),
                                 std::make_shared<Int64Imm>(static_cast<int64_t>(cos->data_type()))),
                     cos, std::make_shared<Int64Imm>(1));
  auto neg = clamp_min(sub_scalar(cos, margin, std::make_shared<Int64Imm>(1)), std::make_shared<Int64Imm>(0));
  auto output_pos = select(eq_scalar(target_tensor, std::make_shared<Int64Imm>(1)), pos, zeros);
  auto output_neg = select(eq_scalar(target_tensor, std::make_shared<Int64Imm>(-1)), neg, zeros);
  auto output = add_ext(output_pos, output_neg, std::make_shared<Int64Imm>(1));

  if (reduction_imm == Reduction::MEAN) {
    output = mean_ext(output, std::nullopt, std::make_shared<BoolImm>(False), std::nullopt);
  } else if (reduction_imm == Reduction::REDUCTION_SUM) {
    output = sum_ext(output, std::nullopt, std::make_shared<BoolImm>(False), std::nullopt);
  }
  op->set_outputs({output});
  MS_LOG(DEBUG) << "CosineEmbeddingLoss Launch end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
