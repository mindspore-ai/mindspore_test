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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ops/base_operator.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace ops {
class OPS_API EnvironDestroyAllInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: a tensor
    return abstract::kNoShape;
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return kBool;
  }
};

class OPS_API EnvironDestroyAll : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EnvironDestroyAll);
  /// \brief Constructor.
  EnvironDestroyAll() : BaseOperator("EnvironDestroyAll") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EnvironDestroyAll, prim::kPrimEnvironDestroyAll, EnvironDestroyAllInfer, false);
}  // namespace ops
}  // namespace mindspore
