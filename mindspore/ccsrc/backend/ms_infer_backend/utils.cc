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

#include <map>
#include <string>

#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_n.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

#include "backend/ms_infer_backend/utils.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

static std::map<std::string, da::ops::Op> primitive_op_name_map = {
  {ops::kNameAdd, da::ops::Op_add},
  {ops::kNameSub, da::ops::Op_sub},
  {ops::kNameMul, da::ops::Op_mul},
  {ops::kNameDiv, da::ops::Op_div},
  {ops::kNameMatMul, da::ops::Op_matmul},
  {ops::kNameNorm, da::ops::Op_norm},
  {ops::kNameReLU, da::ops::Op_relu},
  {ops::kNameGeLU, da::ops::Op_gelu},
  {ops::kNameSiLU, da::ops::Op_silu},
  {ops::kNameSoftmax, da::ops::Op_softmax},
  {ops::kNameBatchMatMul, da::ops::Op_batch_matmul},
  {ops::kNameCast, da::ops::Op_cast},
  {ops::kNameConcat, da::ops::Op_concat},
  {ops::kNameExpandDims, da::ops::Op_expand_dims},
  {ops::kNameGather, da::ops::Op_gather},
  {ops::kNameNeg, da::ops::Op_neg},
  {ops::kNameNotEqual, da::ops::Op_not_equal},
  {ops::kNameReduceMean, da::ops::Op_reduce_mean},
  {ops::kNameReshape, da::ops::Op_reshape},
  {ops::kNameRsqrt, da::ops::Op_rsqrt},
  {ops::kNameSigmoid, da::ops::Op_sigmoid},
  {ops::kNameSquare, da::ops::Op_square},
  {ops::kNameStridedSlice, da::ops::Op_strided_slice},
  {ops::kNameTile, da::ops::Op_tile},
  {ops::kNameTranspose, da::ops::Op_transpose},
  {kMakeTupleOpName, da::ops::Op_make_tuple},
  {kUpdateStateOpName, da::ops::Op_update_state},
  {kLoadOpName, da::ops::Op_load},
  {kDependOpName, da::ops::Op_depend},
  {kReturnOpName, da::ops::Op_return},
};
static std::map<da::ops::Op, const PrimitivePtr> op_primitive_map = {
  {da::ops::Op_add, prim::kPrimAdd},
  {da::ops::Op_sub, prim::kPrimSub},
  {da::ops::Op_mul, prim::kPrimMul},
  {da::ops::Op_div, prim::kPrimDiv},
  {da::ops::Op_matmul, prim::kPrimMatMul},
  {da::ops::Op_norm, prim::kPrimNorm},
  {da::ops::Op_relu, prim::kPrimReLU},
  {da::ops::Op_gelu, prim::kPrimGeLU},
  {da::ops::Op_silu, prim::kPrimSiLU},
  {da::ops::Op_softmax, prim::kPrimSoftmax},
  {da::ops::Op_batch_matmul, prim::kPrimBatchMatMul},
  {da::ops::Op_cast, prim::kPrimCast},
  {da::ops::Op_concat, prim::kPrimConcat},
  {da::ops::Op_expand_dims, prim::kPrimExpandDims},
  {da::ops::Op_gather, prim::kPrimGather},
  {da::ops::Op_neg, prim::kPrimNeg},
  {da::ops::Op_not_equal, prim::kPrimNotEqual},
  {da::ops::Op_reduce_mean, prim::kPrimReduceMean},
  {da::ops::Op_reshape, prim::kPrimReshape},
  {da::ops::Op_rsqrt, prim::kPrimRsqrt},
  {da::ops::Op_sigmoid, prim::kPrimSigmoid},
  {da::ops::Op_square, prim::kPrimSquare},
  {da::ops::Op_strided_slice, prim::kPrimStridedSlice},
  {da::ops::Op_tile, prim::kPrimTile},
  {da::ops::Op_transpose, prim::kPrimTranspose},
  {da::ops::Op_make_tuple, prim::kPrimMakeTuple},
  {da::ops::Op_update_state, prim::kPrimUpdateState},
  {da::ops::Op_load, prim::kPrimLoad},
  {da::ops::Op_depend, prim::kPrimDepend},
  {da::ops::Op_return, prim::kPrimReturn},
};

da::ops::Op ConvertPrimitiveOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = primitive_op_name_map.find(prim->name());
  if (iter != primitive_op_name_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected MS Primitive " << prim->name();
  }
}

const PrimitivePtr ConvertPrimitiveOp(da::ops::Op op) {
  auto iter = op_primitive_map.find(op);
  if (iter != op_primitive_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected DA Op " << op;
  }
}

static std::map<TypeId, da::tensor::Type> type_id_dtype_map = {
  {kNumberTypeBool, da::tensor::Type_Bool},   {kNumberTypeInt16, da::tensor::Type_I16},
  {kNumberTypeInt32, da::tensor::Type_I32},   {kNumberTypeInt64, da::tensor::Type_I64},
  {kNumberTypeFloat16, da::tensor::Type_F16}, {kNumberTypeFloat32, da::tensor::Type_F32},
  {kNumberTypeFloat64, da::tensor::Type_F64}, {kNumberTypeBFloat16, da::tensor::Type_BF16},
  {kObjectTypeMonad, da::tensor::Type_Monad}, {kObjectTypeTuple, da::tensor::Type_Tuple},
  {kMetaTypeNone, da::tensor::Type_None},
};

static std::map<da::tensor::Type, TypeId> dtype_type_id_map = {
  {da::tensor::Type_Bool, kNumberTypeBool},   {da::tensor::Type_I16, kNumberTypeInt16},
  {da::tensor::Type_I32, kNumberTypeInt32},   {da::tensor::Type_I64, kNumberTypeInt64},
  {da::tensor::Type_F16, kNumberTypeFloat16}, {da::tensor::Type_F32, kNumberTypeFloat32},
  {da::tensor::Type_F64, kNumberTypeFloat64}, {da::tensor::Type_BF16, kNumberTypeBFloat16},
  {da::tensor::Type_Monad, kObjectTypeMonad}, {da::tensor::Type_Tuple, kObjectTypeTuple},
  {da::tensor::Type_None, kMetaTypeNone},
};

da::tensor::Type ConvertDataType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);

  auto iter = type_id_dtype_map.find(type->type_id());
  if (iter != type_id_dtype_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected MS Type " << type->type_name();
  }
}

TypeId ConvertDataType(da::tensor::Type dtype) {
  auto iter = dtype_type_id_map.find(dtype);
  if (iter != dtype_type_id_map.end()) {
    return iter->second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected DA Type " << dtype;
  }
}

void SetTensorShape(da::tensor::DATensor *tensor, const ShapeVector &shape_vector) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_CHECK_FAIL(shape_vector.size() < DA_TENSOR_MAX_DIM, "Tensor dimension too large");

  tensor->dim = shape_vector.size();
  for (size_t i = 0; i < tensor->dim; i++) {
    tensor->shape[i] = shape_vector[i];
  }
  tensor->shape[tensor->dim] = 0;
}

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
