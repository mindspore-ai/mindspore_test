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

#include "nnacl/infer/custom_is_inf_infer.h"
#include "nnacl/infer/infer_register.h"

int CustomIsInfInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, C1NUM, C1NUM);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[FIRST_INPUT];
  TensorC *output = outputs[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_NULL_RETURN_ERR(output);
  output->data_type_ = kNumberTypeBool;
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(output, input);
  return NNACL_OK;
}

REG_INFER(CustomIsInf, PrimType_Inner_CustomIsInf, CustomIsInfInferShape)
