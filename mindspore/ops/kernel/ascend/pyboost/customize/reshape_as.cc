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

#include "kernel/ascend/pyboost/customize/reshape_as.h"

#include <algorithm>
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeAsAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                               const BaseTensorPtr &other_tensor) {
  MS_LOG(DEBUG) << "View ReshapeAs Call start";
  std::vector<ValuePtr> shape;
  const ShapeVector &other_shape = other_tensor->shape();
  std::transform(other_shape.begin(), other_shape.end(), std::back_inserter(shape),
                 [](int64_t x) { return MakeValue(x); });
  auto output = reshape(input_tensor, std::make_shared<ValueTuple>(shape));
  MS_LOG(DEBUG) << "View ReshapeAs Call end";
  op->set_outputs({output});
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
