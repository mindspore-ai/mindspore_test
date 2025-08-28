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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYBOOST_CUSTOMIZE_TO_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYBOOST_CUSTOMIZE_TO_H_
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr PYBOOST_API ToDeviceCustomize(const std::shared_ptr<OpRunner> &op,
                                                const mindspore::tensor::TensorPtr &self,
                                                const std::optional<mindspore::Int64ImmPtr> &device,
                                                const std::optional<mindspore::Int64ImmPtr> &dtype,
                                                const mindspore::BoolImmPtr &non_blocking,
                                                const mindspore::BoolImmPtr &copy);

tensor::TensorPtr PYBOOST_API ToDtypeCustomize(const std::shared_ptr<OpRunner> &op,
                                               const mindspore::tensor::TensorPtr &self,
                                               const std::optional<mindspore::Int64ImmPtr> &dtype,
                                               const mindspore::BoolImmPtr &non_blocking,
                                               const mindspore::BoolImmPtr &copy);

tensor::TensorPtr PYBOOST_API ToOtherCustomize(const std::shared_ptr<OpRunner> &op,
                                               const mindspore::tensor::TensorPtr &self,
                                               const mindspore::tensor::TensorPtr &other,
                                               const mindspore::BoolImmPtr &non_blocking,
                                               const mindspore::BoolImmPtr &copy);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYBOOST_CUSTOMIZE_TO_H_
