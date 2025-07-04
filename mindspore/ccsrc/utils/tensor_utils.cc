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

#include "include/common/utils/tensor_utils.h"

namespace mindspore {
namespace tensor {
void SetPromise(const std::tuple<stub::StubNodePtr> &promises, const TensorPtr &tensor) {
  const auto &p = std::get<0>(promises);
  p->SetValue(tensor);
}

void FlattenOutputs(const ValuePtr &value, std::vector<TensorPtr> *outputs) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<Tensor>()) {
    outputs->emplace_back(value->cast<TensorPtr>());
  } else if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    const auto &elements = seq->value();
    for (const auto &element : elements) {
      FlattenOutputs(element, outputs);
    }
  } else {
    MS_LOG(EXCEPTION) << "Not support type " << value->ToString();
  }
}
}  // namespace tensor
}  // namespace mindspore
