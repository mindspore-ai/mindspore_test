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

#include "infer/ops_func_impl/multi_scale_deformable_attn_grad.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int GetDimOfFourTensors(const InferInfoPtr &tensor1, const InferInfoPtr &tensor2, const InferInfoPtr &tensor3,
                        const InferInfoPtr &tensor4, const int dim) {
  ShapeVector tensor1_shape = tensor1->GetShape();
  if (!tensor1->IsDynamicRank() && tensor1_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor1_shape[dim];
  }

  ShapeVector tensor2_shape = tensor2->GetShape();
  if (!tensor2->IsDynamicRank() && tensor2_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor2_shape[dim];
  }

  ShapeVector tensor3_shape = tensor3->GetShape();
  if (!tensor3->IsDynamicRank() && tensor3_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor3_shape[dim];
  }

  ShapeVector tensor4_shape = tensor4->GetShape();
  if (!tensor4->IsDynamicRank() && tensor4_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor4_shape[dim];
  }

  return abstract::Shape::kShapeDimAny;
}

int GetDimOfThreeTensors(const InferInfoPtr &tensor1, const InferInfoPtr &tensor2, const InferInfoPtr &tensor3,
                         const int dim) {
  ShapeVector tensor1_shape = tensor1->GetShape();
  if (!tensor1->IsDynamicRank() && tensor1_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor1_shape[dim];
  }

  ShapeVector tensor2_shape = tensor2->GetShape();
  if (!tensor2->IsDynamicRank() && tensor2_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor2_shape[dim];
  }

  ShapeVector tensor3_shape = tensor3->GetShape();
  if (!tensor3->IsDynamicRank() && tensor3_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor3_shape[dim];
  }

  return abstract::Shape::kShapeDimAny;
}

int GetDimOfTwoTensors(const InferInfoPtr &tensor1, const InferInfoPtr &tensor2, const int dim) {
  ShapeVector tensor1_shape = tensor1->GetShape();
  if (!tensor1->IsDynamicRank() && tensor1_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor1_shape[dim];
  }

  ShapeVector tensor2_shape = tensor2->GetShape();
  if (!tensor2->IsDynamicRank() && tensor2_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor2_shape[dim];
  }

  return abstract::Shape::kShapeDimAny;
}

int GetDimOfTensor(const InferInfoPtr &tensor, const int dim) {
  ShapeVector tensor_shape = tensor->GetShape();
  if (!tensor->IsDynamicRank() && tensor_shape[dim] != abstract::Shape::kShapeDimAny) {
    return tensor_shape[dim];
  }
  return abstract::Shape::kShapeDimAny;
}

ShapeVector GetFirstShape(const InferInfoPtrList &input_infos) {
  int shape_dim0 = 0;
  int shape_dim1 = 1;
  int shape_dim2 = 2;
  int shape_dim3 = 3;

  // Get 0 dim
  int ret_dim0 = GetDimOfFourTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4],
                                     input_infos[kIndex5], shape_dim0);

  // Get 1 dim
  int ret_dim1 = GetDimOfTensor(input_infos[kIndex0], shape_dim1);

  // Get 2 dim
  int ret_dim2 = GetDimOfThreeTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim2);
  if (ret_dim2 == abstract::Shape::kShapeDimAny) {
    int grad_dim2 = GetDimOfTensor(input_infos[kIndex5], shape_dim2);
    int value_dim3 = GetDimOfTensor(input_infos[kIndex0], shape_dim3);
    if (grad_dim2 != abstract::Shape::kShapeDimAny && value_dim3 != abstract::Shape::kShapeDimAny) {
      ret_dim2 = grad_dim2 / value_dim3;
    }
  }

  // Get 3 dim
  int ret_dim3 = GetDimOfTensor(input_infos[kIndex0], shape_dim3);
  if (ret_dim3 == abstract::Shape::kShapeDimAny) {
    int grad_dim2 = GetDimOfTensor(input_infos[kIndex5], shape_dim2);
    int vlw_dim2 = GetDimOfThreeTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim2);
    if (grad_dim2 != abstract::Shape::kShapeDimAny && vlw_dim2 != abstract::Shape::kShapeDimAny) {
      ret_dim3 = grad_dim2 / vlw_dim2;
    }
  }

  return {ret_dim0, ret_dim1, ret_dim2, ret_dim3};
}

ShapeVector GetSecondShape(const InferInfoPtrList &input_infos) {
  int ret_dim5 = 2;

  int shape_dim0 = 0;
  int shape_dim1 = 1;
  int shape_dim2 = 2;
  int shape_dim3 = 3;
  int shape_dim4 = 4;

  // Get 0 dim
  int ret_dim0 = GetDimOfFourTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4],
                                     input_infos[kIndex5], shape_dim0);

  // Get 1 dim
  int ret_dim1 = GetDimOfThreeTensors(input_infos[kIndex3], input_infos[kIndex4], input_infos[kIndex5], shape_dim1);

  // Get 2 dim
  int ret_dim2 = GetDimOfThreeTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim2);
  if (ret_dim2 == abstract::Shape::kShapeDimAny) {
    int grad_dim2 = GetDimOfTensor(input_infos[kIndex5], shape_dim2);
    int val_dim3 = GetDimOfTensor(input_infos[kIndex0], shape_dim3);
    if (grad_dim2 != abstract::Shape::kShapeDimAny && val_dim3 != abstract::Shape::kShapeDimAny) {
      ret_dim2 = grad_dim2 / val_dim3;
    }
  }

  // Get 3 dim
  int ret_dim3 = GetDimOfTwoTensors(input_infos[kIndex3], input_infos[kIndex4], shape_dim3);
  if (ret_dim3 == abstract::Shape::kShapeDimAny) {
    ret_dim3 = GetDimOfTwoTensors(input_infos[kIndex1], input_infos[kIndex2], shape_dim0);
  }

  // Get 4 dim
  int ret_dim4 = GetDimOfTwoTensors(input_infos[kIndex3], input_infos[kIndex4], shape_dim4);

  return {ret_dim0, ret_dim1, ret_dim2, ret_dim3, ret_dim4, ret_dim5};
}

ShapeVector GetThirdShape(const InferInfoPtrList &input_infos) {
  int shape_dim0 = 0;
  int shape_dim1 = 1;
  int shape_dim2 = 2;
  int shape_dim3 = 3;
  int shape_dim4 = 4;

  // Get 0 dim
  int ret_dim0 = GetDimOfFourTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4],
                                     input_infos[kIndex5], shape_dim0);

  // Get 1 dim
  int ret_dim1 = GetDimOfThreeTensors(input_infos[kIndex3], input_infos[kIndex4], input_infos[kIndex5], shape_dim1);

  // Get 2 dim
  int ret_dim2 = GetDimOfThreeTensors(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim2);
  if (ret_dim2 == abstract::Shape::kShapeDimAny) {
    int grad_dim2 = GetDimOfTensor(input_infos[kIndex5], shape_dim2);
    int val_dim3 = GetDimOfTensor(input_infos[kIndex0], shape_dim3);
    if (grad_dim2 != abstract::Shape::kShapeDimAny && val_dim3 != abstract::Shape::kShapeDimAny) {
      ret_dim2 = grad_dim2 / val_dim3;
    }
  }

  // Get 3 dim
  int ret_dim3 = GetDimOfTwoTensors(input_infos[kIndex3], input_infos[kIndex4], shape_dim3);
  if (ret_dim3 == abstract::Shape::kShapeDimAny) {
    ret_dim3 = GetDimOfTwoTensors(input_infos[kIndex1], input_infos[kIndex2], shape_dim0);
  }

  int ret_dim4 = GetDimOfTwoTensors(input_infos[kIndex3], input_infos[kIndex4], shape_dim4);

  return {ret_dim0, ret_dim1, ret_dim2, ret_dim3, ret_dim4};
}
}  // namespace
ShapeArray MultiScaleDeformableAttnGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  ShapeVector shape1 = GetFirstShape(input_infos);
  ShapeVector shape2 = GetSecondShape(input_infos);
  ShapeVector shape3 = GetThirdShape(input_infos);

  return {shape1, shape2, shape3};
}

std::vector<TypeId> MultiScaleDeformableAttnGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                    const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType(), input_infos[3]->GetType(), input_infos[4]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
