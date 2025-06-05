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

#ifndef MINDSPORE_CORE_OPS_OP_FUNC_IMPL_INPLACE_BERNOULLI_SCALAR_H_
#define MINDSPORE_CORE_OPS_OP_FUNC_IMPL_INPLACE_BERNOULLI_SCALAR_H_

#include "infer/ops_func_impl/inplace_bernoulli_tensor.h"

namespace mindspore::ops {
class InplaceBernoulliScalarFuncImpl : public InplaceBernoulliTensorFuncImpl {};
}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_OP_FUNC_IMPL_INPLACE_BERNOULLI_SCALAR_H_
