/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "mindspore/ops/kernel/cpu/pyboost/customize/matmul_ext.h"
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/matmul.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/batch_mat_mul.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/reshape.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/broadcast_to.h"
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/contiguous.h"
#include "infer/ops_func_impl/matmul_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
size_t Rank(const TensorPtr &x) { return x->shape_c().size(); }

TensorPtr Expand(TensorPtr tensor, size_t ndim, const DeviceContext *device_context) {
  auto reshape = CREATE_PYBOOST_OP(Reshape, device_context->device_context_key_.device_name_);
  ShapeVector shape = tensor->shape();
  while (shape.size() < ndim) {
    shape.insert(shape.begin(), 1);
  }
  tensor = reshape->Call(tensor, shape);
  return tensor;
}

std::vector<int64_t> ReduceTo3D(const ShapeVector &shape) {
  ShapeVector ret;
  int64_t dim0 = 1;
  for (size_t i = 0; i < shape.size() - kDim2; ++i) {
    dim0 *= shape[i];
  }
  ret.push_back(dim0);
  ret.push_back(shape[shape.size() - kDim2]);
  ret.push_back(shape[shape.size() - kDim1]);
  return ret;
}

}  // namespace
void MatMulExtCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                           const TensorPtr &mat2_tensor) {
  MS_LOG(DEBUG) << "Call start";
  OpRunner::InferOpOutput(op, input_tensor, mat2_tensor);
  auto device_context = op->device_context();

  // convert input_tensor into input, input is a TensorPtr
  TensorPtr input = input_tensor;
  TensorPtr other = mat2_tensor;

  auto input_rank = input->shape().size();
  auto other_rank = other->shape().size();

  auto matmul = CREATE_PYBOOST_OP(MatMul, device_context->device_context_key_.device_name_);
  auto batch_matmul = CREATE_PYBOOST_OP(BatchMatMul, device_context->device_context_key_.device_name_);

  auto reshape_1 = CREATE_PYBOOST_OP(Reshape, device_context->device_context_key_.device_name_);
  auto reshape_2 = CREATE_PYBOOST_OP(Reshape, device_context->device_context_key_.device_name_);
  auto reshape_3 = CREATE_PYBOOST_OP(Reshape, device_context->device_context_key_.device_name_);
  auto reshape_4 = CREATE_PYBOOST_OP(Reshape, device_context->device_context_key_.device_name_);

  auto contiguous_1 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);
  auto contiguous_2 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);
  auto contiguous_3 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);
  auto contiguous_4 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);
  auto contiguous_5 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);
  auto contiguous_6 = CREATE_PYBOOST_OP(Contiguous, device_context->device_context_key_.device_name_);

  if (input_rank == kDim2 && other_rank == kDim2) {
    matmul->Call(input, other, std::make_shared<BoolImm>(false), std::make_shared<BoolImm>(false));
    op->set_outputs(matmul->outputs());
    MS_LOG(DEBUG) << "Launch end"
                  << "2D";
    return;
  }

  const ShapeVector &shape1_orig = input->shape();
  const ShapeVector &shape2_orig = other->shape();
  bool transpose_b = other_rank == 1;

  ShapeVector shape_backbone = ops::CheckMatMulShapes(shape1_orig, shape2_orig);
  ShapeVector shape_out = ops::InferShapeRem(shape_backbone, shape1_orig, shape2_orig, transpose_b);

  input = Expand(input, kDim2, device_context);
  other = Expand(other, kDim2, device_context);

  TensorPtr res;
  if (Rank(other) == kDim2) {
    if (Rank(input) > kDim2) {
      int64_t new_shape_dim0 = 1;
      for (size_t i = 0; i < shape1_orig.size() - 1; ++i) {
        new_shape_dim0 *= shape1_orig[i];
      }
      std::vector<int64_t> new_shape_vector = {new_shape_dim0, shape1_orig.back()};
      input = contiguous_1->Call(reshape_1->Call(input, new_shape_vector));
    }
    res = matmul->Call(input, other, std::make_shared<BoolImm>(false), std::make_shared<BoolImm>(transpose_b));
  } else {
    int ndim_aligned = std::max(input_rank, other_rank);
    input = Expand(input, ndim_aligned, device_context);
    other = Expand(other, ndim_aligned, device_context);

    ShapeVector shape1_aligned = input->shape();
    ShapeVector shape2_aligned = other->shape();

    ShapeVector shape_cur1(shape1_aligned.begin(), shape1_aligned.end() - kDim2);
    ShapeVector shape_cur2(shape2_aligned.begin(), shape2_aligned.end() - kDim2);

    if (shape_cur1 != shape_backbone) {
      auto broadcast_to = CREATE_PYBOOST_OP(BroadcastTo, device_context->device_context_key_.device_name_);
      input =
        contiguous_5->Call(broadcast_to->Call(input, ops::GetMatMulExtBroadcastShape(shape_backbone, shape1_orig)));
    }

    if (shape_cur2 != shape_backbone) {
      auto broadcast_to = CREATE_PYBOOST_OP(BroadcastTo, device_context->device_context_key_.device_name_);
      other =
        contiguous_6->Call(broadcast_to->Call(other, ops::GetMatMulExtBroadcastShape(shape_backbone, shape2_orig)));
    }

    input = contiguous_3->Call(reshape_3->Call(input, ReduceTo3D(input->shape())));
    other = contiguous_4->Call(reshape_4->Call(other, ReduceTo3D(other->shape())));

    res = batch_matmul->Call(input, other, std::make_shared<BoolImm>(false), std::make_shared<BoolImm>(transpose_b));
  }
  contiguous_2->Call(reshape_2->Call(res, shape_out));
  op->set_outputs(contiguous_2->outputs());
  MS_LOG(DEBUG) << "Launch end"
                << "nD";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
