/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_DYNAMIC_BROADCAST_GRADIENT_ARGS_H_
#define MINDSPORE_CORE_OPS_DYNAMIC_BROADCAST_GRADIENT_ARGS_H_
#include <memory>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace ops {
constexpr auto kNameDynamicBroadcastGradientArgs = "DynamicBroadcastGradientArgs";
class OPS_API DynamicBroadcastGradientArgs : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicBroadcastGradientArgs);
  DynamicBroadcastGradientArgs() : BaseOperator(kNameDynamicBroadcastGradientArgs) {}
  void Init() const {}
};

OPS_API ShapeArray BroadcastGradientArgsInferValue(const ShapeVector &x_shape, const ShapeVector &y_shape,
                                                   size_t ignore_offset = 0UL);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DYNAMIC_BROADCAST_GRADIENT_ARGS_H_
