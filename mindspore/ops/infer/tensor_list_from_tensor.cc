/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "infer/tensor_list_from_tensor.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {
void TensorListFromTensor::Init(const int64_t element_dtype, const int64_t shape_type) {
  this->set_element_dtype(element_dtype);
  this->set_shape_type(shape_type);
}

int64_t TensorListFromTensor::get_element_dtype() const {
  auto value_ptr = GetAttr(kElement_dtype);
  return GetValue<int64_t>(value_ptr);
}

int64_t TensorListFromTensor::get_shape_type() const {
  auto value_ptr = GetAttr(kShapeType);
  return GetValue<int64_t>(value_ptr);
}

void TensorListFromTensor::set_element_dtype(const int64_t element_dtype) {
  (void)this->AddAttr(kElement_dtype, api::MakeValue(element_dtype));
}

void TensorListFromTensor::set_shape_type(const int64_t shape_type) {
  (void)this->AddAttr(kShapeType, api::MakeValue(shape_type));
}

MIND_API_OPERATOR_IMPL(TensorListFromTensor, BaseOperator);
REGISTER_PRIMITIVE_C(kNameTensorListFromTensor, TensorListFromTensor);
}  // namespace ops
}  // namespace mindspore
