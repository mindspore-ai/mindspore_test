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

#ifndef MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_ACLNN_LAUNCH_H_
#define MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_ACLNN_LAUNCH_H_

#include <string>
#include "ir/tensor.h"

namespace mindspore::custom {
__attribute__((visibility("default"))) void CustomLaunchAclnn(const std::string &aclnn_api, const ValuePtrList &inputs,
                                                              const tensor::TensorPtrList &outputs);
}  // namespace mindspore::custom
#endif  // MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_CUSTOM_ACLNN_LAUNCH_H_
