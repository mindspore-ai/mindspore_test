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

#include "infer/ops_func_impl/ge_graph_op.h"
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
BaseShapePtr GEGraphOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  if (!primitive->HasAttr("output_num")) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the attr output_num is not set.";
  }
  auto output_num = GetScalarValue<int64_t>(primitive->GetAttr("output_num"));

  // for GEGraphOp in dynamic shape, set output shape as 0, it will update after RunGraph
  std::vector<abstract::BaseShapePtr> output_shape;
  for (auto i = 0; i < output_num; ++i) {
    output_shape.emplace_back(std::make_shared<abstract::Shape>(ShapeVector{0}));
  }

  return std::make_shared<abstract::TupleShape>(output_shape);
}
}  // namespace mindspore::ops
