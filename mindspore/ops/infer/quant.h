/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_QUANT_H_
#define MINDSPORE_CORE_OPS_QUANT_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameQuant = "Quant";
class OPS_API Quant : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Quant);
  /// \brief Constructor.
  Quant() : BaseOperator(kNameQuant) { InitIOName({"x"}, {"output"}); }
  explicit Quant(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};

OPS_API abstract::AbstractBasePtr QuantInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_QUANT_H_
