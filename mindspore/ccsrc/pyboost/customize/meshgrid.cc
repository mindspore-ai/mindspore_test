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

#include "mindspore/ccsrc/pyboost/customize/meshgrid.h"
#include <memory>
#include <utility>
#include <string>
#include "runtime/device/device_address_utils.h"
#include "kernel/ascend/pyboost/auto_generate/view.h"
#include "kernel/ascend/pyboost/auto_generate/broadcast_to.h"
#include "op_def/op_enum.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "utils/core_op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

std::vector<tensor::BaseTensorPtr> MeshgridCustomizeCall(const std::shared_ptr<OpRunner> &op,
                                                         const ValueTuplePtr &tensors_list, const Int64ImmPtr &indexing,
                                                         const string &device_type) {
  MS_LOG(DEBUG) << "Meshgrid call start";
  std::vector<BaseTensorPtr> tensors_list_vector;

  const auto &tensors_value = tensors_list->value();
  MS_CHECK_VALUE(tensors_value.size() > 0,
                 "For Primitive [Meshgrid], the size of input tensors must be greater than 0.");
  for (const auto &tensor : tensors_value) {
    if (tensor->isa<BaseTensor>()) {
      (void)tensors_list_vector.emplace_back(GetValue<BaseTensorPtr>(tensor));
    } else if (tensor->isa<Scalar>()) {
      (void)tensors_list_vector.emplace_back(PyBoostUtils::ScalarToTensor(tensor->cast<ScalarPtr>()));
    }
  }

  for (size_t i = 0; i < tensors_list_vector.size() - 1; ++i) {
    MS_CHECK_VALUE(tensors_list_vector[i]->data_type() == tensors_list_vector[i + 1]->data_type(),
                   "For Primitive [Meshgrid], all tensors should have the same type.");
  }

  ops::Indexing indexing_imm = static_cast<ops::Indexing>(GetValue<int64_t>(indexing));

  bool swap_tensors = false;
  const size_t MIN_SWAP_SIZE = 2;
  if (indexing_imm == ops::Indexing::XY && tensors_list_vector.size() >= MIN_SWAP_SIZE) {
    swap_tensors = true;
    std::swap(tensors_list_vector[kIndex0], tensors_list_vector[kIndex1]);
  }

  std::vector<tensor::BaseTensorPtr> view_outputs;
  auto view_shape_list = std::vector<ValuePtr>(tensors_list_vector.size(), std::make_shared<Int64Imm>(1));
  for (size_t i = 0; i < tensors_list_vector.size(); ++i) {
    view_shape_list[i] = std::make_shared<Int64Imm>(-1);
    auto view_op = CREATE_PYBOOST_OP(View, device_type);
    view_op->Call(tensors_list_vector[i], std::make_shared<ValueTuple>(view_shape_list));
    view_outputs.push_back(view_op->outputs()[kIndex0]);
    view_shape_list[i] = std::make_shared<Int64Imm>(1);
  }

  std::vector<ValuePtr> output_shape_list;
  const int64_t SCALAR_TO_TENSOR_SIZE = 1;
  //   ShapeVector outputs_shape;
  for (auto tensor : tensors_list_vector) {
    const auto &input_shape = tensor->shape();
    if (input_shape.empty()) {
      output_shape_list.push_back(std::make_shared<Int64Imm>(SCALAR_TO_TENSOR_SIZE));
    } else {
      output_shape_list.push_back(std::make_shared<Int64Imm>(input_shape[kIndex0]));
    }
  }

  std::vector<tensor::BaseTensorPtr> broadcast_to_outputs;
  for (auto view_tensor : view_outputs) {
    auto broadcast_to_op = CREATE_PYBOOST_OP(BroadcastTo, device_type);
    broadcast_to_op->Call(view_tensor, std::make_shared<ValueTuple>(output_shape_list));
    broadcast_to_outputs.push_back(broadcast_to_op->outputs()[kIndex0]);
  }

  if (swap_tensors) {
    std::swap(broadcast_to_outputs[kIndex0], broadcast_to_outputs[kIndex1]);
  }

  op->set_outputs(broadcast_to_outputs);
  MS_LOG(DEBUG) << "Meshgrid call end";

  return broadcast_to_outputs;
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
