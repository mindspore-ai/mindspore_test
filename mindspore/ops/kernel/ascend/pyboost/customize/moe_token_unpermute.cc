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

#include "kernel/ascend/pyboost/customize/moe_token_unpermute.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "mindspore/ops/ops_utils/op_constants.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr MoeTokenUnpermuteAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                   const TensorPtr &permuted_tokens, const TensorPtr &sorted_indices,
                                                   const std::optional<TensorPtr> &probs, const BoolImmPtr &padded_mode,
                                                   const std::optional<ValueTuplePtr> &restore_shape) {
  if (probs.has_value()) {
    auto target_dtype = probs.value()->data_type();
    if (target_dtype == kNumberTypeFloat32) {
      auto origin_dtype = permuted_tokens->data_type();
      auto unpermute_token_casted = cast(permuted_tokens, std::make_shared<Int64Imm>(target_dtype));
      auto out = inner_moe_token_unpermute(unpermute_token_casted, sorted_indices, probs, padded_mode, restore_shape);
      out = cast(out, std::make_shared<Int64Imm>(origin_dtype));
      op->set_outputs({out});
      return op->output(0);
    }
  }
  auto out = inner_moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
  op->set_outputs({out});
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
