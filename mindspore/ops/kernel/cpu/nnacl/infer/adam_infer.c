/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "nnacl/infer/adam_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensor_c_utils.h"

int AdamInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                   OpParameter *parameter) {
  int check_ret = CheckAugmentNullInputSize(inputs, inputs_size, outputs, outputs_size, parameter, 10);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  if (NNACLGetElementNum(inputs[0]) != NNACLGetElementNum(inputs[1]) ||
      NNACLGetElementNum(inputs[0]) != NNACLGetElementNum(inputs[2]) ||
      NNACLGetElementNum(inputs[0]) != NNACLGetElementNum(inputs[9]) || NNACLGetElementNum(inputs[3]) != 1 ||
      NNACLGetElementNum(inputs[4]) != 1 || NNACLGetElementNum(inputs[5]) != 1 || NNACLGetElementNum(inputs[6]) != 1 ||
      NNACLGetElementNum(inputs[7]) != 1 || NNACLGetElementNum(inputs[8]) != 1) {
    return NNACL_ERR;
  }
  if (outputs_size != 0) {
    TensorC *out = outputs[0];
    SetDataTypeFormat(out, inputs[0]);
    out->shape_size_ = 1;
    out->shape_[0] = 1;
  }

  return NNACL_OK;
}

REG_INFER(Adam, PrimType_Adam, AdamInferShape)
