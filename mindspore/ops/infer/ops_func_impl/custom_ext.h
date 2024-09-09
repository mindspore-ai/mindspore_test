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

#ifndef MINDSPORE_OPS_INFER_FUNC_IMPL_CUSTOM_EXT_H_
#define MINDSPORE_OPS_INFER_FUNC_IMPL_CUSTOM_EXT_H_

#include <memory>
#include <vector>
#include <set>
#include "mindapi/base/macros.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore::ops {
using InferShapeCallback =
  std::function<BaseShapePtr(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args)>;

using InferTypeCallback =
  std::function<TypePtr(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args)>;

class OPS_API CustomExtFuncImpl : public OpFuncImpl {
 public:
  static void set_infer_shape_call_func(const InferShapeCallback &call) { infer_shape_func_ = call; }
  static void set_infer_type_call_func(const InferTypeCallback &call) { infer_type_func_ = call; }
  CustomExtFuncImpl() = default;
  ~CustomExtFuncImpl() = default;

  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

 private:
  static InferShapeCallback infer_shape_func_;
  static InferTypeCallback infer_type_func_;
};

using CustomFuncImplPtr = std::shared_ptr<CustomExtFuncImpl>;
}  // namespace mindspore::ops
#endif  // MINDSPORE_OPS_INFER_FUNC_IMPL_CUSTOM_EXT_H_
