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

#include "ops/ops_frontend_func_impl.h"

namespace mindspore::ops {
namespace {
constexpr auto kDivMods = "DivMods";
}  // namespace

class DivModsFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    return InferValueCallback::GetInstance().CallPyInferValue(kDivMods, input_args);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL(kDivMods, DivModsFrontendFuncImpl);
}  // namespace mindspore::ops
