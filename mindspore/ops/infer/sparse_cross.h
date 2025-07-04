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

#ifndef MINDSPORE_CORE_OPS_SPARSE_CROSS_H_
#define MINDSPORE_CORE_OPS_SPARSE_CROSS_H_
#include <memory>
#include <set>
#include <vector>
#include "abstract/abstract_value.h"
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace ops {
constexpr auto kNameSparseCross = "SparseCross";
class OPS_API SparseCross : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(SparseCross);
  SparseCross() : BaseOperator(kNameSparseCross) {
    InitIOName({"indices", "values", "shapes", "dense_inputs"}, {"output_indices", "output_values", "output_shape"});
  }
  void Init() const {}
};

AbstractBasePtr SparseCrossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args);
using kSparseCrossPtr = std::shared_ptr<SparseCross>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPARSE_CROSS_H_
