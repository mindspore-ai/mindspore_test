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

#include "kernel/ascend/pyboost/customize/index.h"
#include "kernel/ascend/pyboost/auto_generate/inner_non_zero.h"
#include "kernel/ascend/pyboost/auto_generate/select_ext.h"
#include "kernel/ascend/pyboost/auto_generate/inner_index.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<BaseTensorPtr> GetNewTensor(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                        const std::vector<BaseTensorPtr> &tensors) {
  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;
  std::vector<BaseTensorPtr> result{};
  auto input_shape = input_tensor->shape();
  if (input_shape.size() == 0) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << input_shape.size();
  }
  if (tensors.size() > input_shape.size()) {
    MS_EXCEPTION(ValueError) << "For 'Index', too many indices for tensor of dimension " << input_shape.size()
                             << " (got " << tensors.size() << ")";
  }
  bool needCast = false;
  TypeId indicesDtype = tensors[0]->data_type();
  for (const auto &tensor : tensors) {
    auto type_id = tensor->data_type();
    if (type_id != kNumberTypeInt64 && type_id != kNumberTypeInt32 && type_id != kNumberTypeBool &&
        type_id != kNumberTypeUInt8) {
      MS_EXCEPTION(TypeError) << "For 'Index', tensors used as indices must be long, int, uint8, or bool tensors";
    }
    if (type_id == kNumberTypeBool || type_id == kNumberTypeUInt8) {
      auto shape = tensor->shape();
      auto rank = SizeToLong(shape.size());
      for (int64_t j = 0; j < rank; j++) {
        auto srcIdx = result.size() + j;
        if (shape[j] != input_shape[srcIdx]) {
          MS_EXCEPTION(ValueError) << "For 'Index', the shape of the mask " << tensor->ElementsNum() << " at index "
                                   << j << " does not match the shape of the indexed tensor " << input_shape
                                   << " at index " << srcIdx;
        }
      }
      auto nonzero_op = CREATE_PYBOOST_OP(InnerNonZero, device_name);
      auto nonzero_tensor = nonzero_op->Call(tensor);
      for (int64_t j = 0; j < rank; j++) {
        const auto dim = std::make_shared<Int64Imm>(kIndex0);
        const auto index = std::make_shared<Int64Imm>(j);
        auto select_op = CREATE_PYBOOST_OP(SelectExt, device_name);
        auto select_tensor = select_op->Call(nonzero_tensor, dim, index);
        result.emplace_back(select_tensor);
      }
    } else {
      result.emplace_back(tensor);
    }
    if (indicesDtype != type_id) {
      needCast = true;
    }
  }
  if (needCast) {
    for (size_t i = 0; i < result.size(); i++) {
      if (result[i]->data_type() == kNumberTypeInt32) {
        result[i] = PyBoostUtils::CastTensor(result[i], kNumberTypeInt64, device_name);
      }
    }
  }
  return result;
}
}  // namespace

tensor::BaseTensorPtr IndexAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                           const ValueTuplePtr &indices_tensor_list) {
  MS_LOG(DEBUG) << "Index Ascend start";
  std::vector<BaseTensorPtr> indices_tensor_vector = ConvertValueTupleToVector<BaseTensorPtr>(indices_tensor_list);
  auto new_indices_tensor_vector = GetNewTensor(op, input_tensor, indices_tensor_vector);
  ValueTuplePtr new_indices_tensor_list = PyBoostUtils::ConvertTensorVectorToTuple(new_indices_tensor_vector);

  auto device_context = op->device_context();
  const auto &device_name = device_context->device_context_key_.device_name_;
  auto inner_index_op = CREATE_PYBOOST_OP(InnerIndex, device_name);
  auto index_out = inner_index_op->Call(input_tensor, new_indices_tensor_list);
  op->set_outputs(inner_index_op->outputs());
  MS_LOG(DEBUG) << "Index Ascend end";
  return index_out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
