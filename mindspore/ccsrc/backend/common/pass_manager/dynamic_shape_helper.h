/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H
#define MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H

#include <vector>
#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/functor.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/infershape_functor.h"

namespace mindspore::opt::dynamic_shape {
BACKEND_COMMON_EXPORT BaseShapePtr InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args);

BACKEND_COMMON_EXPORT void UpdateKernelTensorShape(const BaseShapePtr &base_shape,
                                                   const std::vector<kernel::KernelTensor *> &output_kernel_tensors);

BACKEND_COMMON_EXPORT abstract::AbstractBasePtr InferShapeAndType(const PrimitivePtr &primitive,
                                                                  const std::vector<AbstractBasePtr> &input_args);

BACKEND_COMMON_EXPORT void UpdateKernelTensorType(const TypePtr &type,
                                                  const std::vector<kernel::KernelTensor *> &output_kernel_tensors);
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_OPTIMIZER_DYNAMIC_SHAPE_DYNAMIC_SHAPE_HELPER_H
