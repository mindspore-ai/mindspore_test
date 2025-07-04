/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_RESHAPE_EXT_H_
#define MINDSPORE_CORE_OPS_RESHAPE_EXT_H_
#include <map>
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReshapeExt = "ReshapeExt";

/// \brief ReshapeExt.
/// Refer to Python API @ref mindspore.ops.ReshapeExt for more details.
class OPS_API ReshapeExt : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReshapeExt);
  /// \brief Constructor.
  ReshapeExt() : BaseOperator(kNameReshapeExt) {}
};

AbstractBasePtr ReshapeExtInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args);
using ReshapeExtPtr = std::shared_ptr<ReshapeExt>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RESHAPE_EXT_H_
