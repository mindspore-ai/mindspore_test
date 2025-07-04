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

#ifndef MINDSPORE_CORE_OPS_SETSIZE_H_
#define MINDSPORE_CORE_OPS_SETSIZE_H_

#include <memory>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSetSize = "SetSize";
class OPS_API SetSize : public BaseOperator {
 public:
  SetSize() : BaseOperator(kNameSetSize) { InitIOName({"set_indices", "set_values", "set_shape"}, {"size"}); }
  void Init(const bool validate_indices = true);
  MIND_API_BASE_MEMBER(SetSize);
  void set_validate_indices(const bool &validate_indices);

  bool get_validate_indices() const;
};
OPS_API abstract::AbstractBasePtr SetSizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SETSIZE_H_
