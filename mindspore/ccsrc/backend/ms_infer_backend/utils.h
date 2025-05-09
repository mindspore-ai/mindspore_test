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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_UTILS_H_

#include "ir/primitive.h"

#include "dalang/dair/ops/ops_name.h"
#include "dalang/dair/tensor/tensor.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

// Convert Primitive to Op
da::ops::Op ConvertPrimitiveOp(const PrimitivePtr &prim);

// Convert Op to Primitive
const PrimitivePtr ConvertPrimitiveOp(const da::ops::Op op);

// Convert TypePtr to da::tensor::Type
da::tensor::Type ConvertDataType(const TypePtr &type);

// Convert da::tensor::Type to TypeId
TypeId ConvertDataType(da::tensor::Type dtype);

// Set the shape vector to a DATensor
void SetTensorShape(da::tensor::DATensor *tensor, const ShapeVector &shape_vector);

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_UTILS_H_
