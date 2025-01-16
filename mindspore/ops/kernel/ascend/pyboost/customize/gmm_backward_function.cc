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

#include "kernel/ascend/pyboost/customize/gmm_backward.h"
#include "kernel/ascend/pyboost/customize/gmm_v2_backward.h"

#include <memory>
#include <functional>
#include <vector>

#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
bool IsTensorTransposed(const ValueTuplePtr &tuple_tensor) {
  const auto &tensors = tuple_tensor->value();
  auto tensor = tensors.at(kIndex0)->cast<BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &tensor_storage_info = tensor->storage_info();
  if (tensor_storage_info == nullptr) {
    return false;
  }
  const auto &shape = tensor_storage_info->shape;
  const auto &strides = tensor_storage_info->strides;
  if (shape.size() < kIndex2 || shape.size() > kIndex3) {
    MS_EXCEPTION(ValueError)
      << "input tensor of func 'IsTensorTransposed' should be either 2- or 3-dimensional, bit got input tensor's rank: "
      << shape.size();
  }
  if (strides[strides.size() - kIndex2] == SizeToLong(kIndex1) && strides.back() == shape[shape.size() - kIndex2]) {
    return true;
  }
  return false;
}

BaseTensorPtr TransposeLastTwoDim(const BaseTensorPtr &tensor) {
  static const auto dim0 = std::make_shared<Int64Imm>(-1);
  static const auto dim1 = std::make_shared<Int64Imm>(-2);
  return transpose_ext(tensor, dim0, dim1);
}

ValueTuplePtr ForEachTranspose(const ValueTuplePtr &tensor_list, bool to_contiguous = false) {
  std::vector<ValuePtr> elements;
  const auto &tensors = tensor_list->value();
  for (const auto &tensor : tensors) {
    auto tensor_i = tensor->cast<BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_i);
    auto contiguous_tensor_i = to_contiguous ? contiguous(tensor_i) : tensor_i;
    (void)elements.emplace_back(TransposeLastTwoDim(contiguous_tensor_i));
  }
  return std::make_shared<ValueTuple>(elements);
}

std::vector<BaseTensorPtr> ForEachTranspose(const std::vector<BaseTensorPtr> &tensors) {
  std::vector<BaseTensorPtr> results;
  for (const auto &tensor : tensors) {
    (void)results.emplace_back(TransposeLastTwoDim(tensor));
  }
  return results;
}

std::vector<BaseTensorPtr> ForEachReShape(const std::vector<BaseTensorPtr> &tensors,
                                          const ValueTuplePtr &target_tensor_list) {
  std::vector<BaseTensorPtr> results;
  const auto &target_tensors = target_tensor_list->value();
  MS_ASSERT(tensors.size() == target_tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto target_tensor_i = target_tensors[i]->cast<BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(target_tensor_i);
    const auto &target_i_shape = MakeValue<std::vector<int64_t>>(target_tensor_i->shape());
    auto tensor_i_new_shape = target_i_shape->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tensor_i_new_shape);
    auto tensor_i_t = reshape(tensors[i], tensor_i_new_shape);
    (void)results.emplace_back(std::move(tensor_i_t));
  }
  return results;
}

std::vector<BaseTensorPtr> Gmm(const ValueTuplePtr &x, const ValueTuplePtr &weight,
                               const std::optional<ValueTuplePtr> &group_list, int64_t group_type_value) {
  static const auto split_item = std::make_shared<Int64Imm>(3);
  const auto group_type = std::make_shared<Int64Imm>(group_type_value);
  return grouped_matmul_v2(x, weight, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, group_list,
                           split_item, group_type);
}

std::vector<BaseTensorPtr> GmmV2(const ValueTuplePtr &x, const ValueTuplePtr &weight,
                                 const std::optional<BaseTensorPtr> &group_list, const Int64ImmPtr &group_list_type,
                                 int64_t group_type_value) {
  static const auto split_item = std::make_shared<Int64Imm>(3);
  const auto group_type = std::make_shared<Int64Imm>(group_type_value);
  static const auto act_type = std::make_shared<Int64Imm>(0);
  return grouped_matmul_v4(x, weight, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                           std::nullopt, group_list, std::nullopt, std::nullopt, std::nullopt, split_item, group_type,
                           group_list_type, act_type);
}
}  // namespace
void GmmBackwardAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &grad_tenor_list,
                                const ValueTuplePtr &x_tensor_list, const ValueTuplePtr &weight_tensor_list,
                                const std::optional<ValueTuplePtr> &group_list) {
  MS_LOG(DEBUG) << "GMMBackward launch start.";

  auto xt = ForEachTranspose(x_tensor_list);
  auto wt = ForEachTranspose(weight_tensor_list);

  auto dx = Gmm(grad_tenor_list, wt, group_list, 0);
  auto dw = Gmm(xt, grad_tenor_list, group_list, 2);
  auto dw_output = ForEachReShape(dw, weight_tensor_list);

  auto &all_gradients = dx;
  all_gradients.insert(all_gradients.end(), dw_output.begin(), dw_output.end());

  op->set_outputs(all_gradients);
  MS_LOG(DEBUG) << "GMMBackward launch end.";
}

void GmmV2BackwardAscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &grad_tenor_list,
                                  const ValueTuplePtr &x_tensor_list, const ValueTuplePtr &weight_tensor_list,
                                  const std::optional<BaseTensorPtr> &group_list, const Int64ImmPtr group_list_type) {
  MS_LOG(DEBUG) << "GMMV2Backward launch start.";

  auto wt = ForEachTranspose(weight_tensor_list);
  auto dx = GmmV2(grad_tenor_list, wt, group_list, group_list_type, 0);

  std::vector<BaseTensorPtr> dw;
  if (IsTensorTransposed(weight_tensor_list)) {
    auto gradt = ForEachTranspose(grad_tenor_list, true);
    auto dwt = GmmV2(gradt, x_tensor_list, group_list, group_list_type, 2);
    dw = ForEachTranspose(dwt);
  } else {
    auto xt = ForEachTranspose(x_tensor_list);
    dw = GmmV2(xt, grad_tenor_list, group_list, group_list_type, 2);
  }
  auto dw_output = ForEachReShape(dw, weight_tensor_list);

  auto &all_gradients = dx;
  all_gradients.insert(all_gradients.end(), dw_output.begin(), dw_output.end());

  op->set_outputs(all_gradients);
  MS_LOG(DEBUG) << "GMMV2Backward launch end.";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
