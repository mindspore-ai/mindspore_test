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

#include "kernel/ascend/aclnn/pyboost_impl/customize/real_view.h"
#include <unordered_map>
namespace mindspore {
namespace kernel {
namespace pyboost {

TypeId GetRealTypeFromComplex(TypeId complex_type) {
  static const std::unordered_map<TypeId, TypeId> complex_to_real_map = {
    {kNumberTypeComplex, kNumberTypeFloat16},     // complex -> float16
    {kNumberTypeComplex64, kNumberTypeFloat32},   // complex64 -> float32
    {kNumberTypeComplex128, kNumberTypeFloat64},  // complex128 -> float64
  };

  auto it = complex_to_real_map.find(complex_type);
  if (it != complex_to_real_map.end()) {
    return it->second;
  }
  // if not found, return unknown
  return kTypeUnknown;
}

tensor::TensorPtr RealImagViewAscendCustomizeBase(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                                  bool is_real_view) {
  MS_LOG(DEBUG) << "View RealView Call start";
  auto input_data_type = input_tensor->data_type();
  TypeId data_type = GetRealTypeFromComplex(input_data_type);
  auto is_complex_data_type = data_type != kTypeUnknown;
  if (!is_real_view && !is_complex_data_type) {  // ImagView
    MS_EXCEPTION(TypeError) << "For primitive [ImagView], "
                            << "the input tensor data type must be complex64 or complex128, "
                            << "but got " << TypeIdToString(input_data_type) << ".";
  }
  data_type = is_complex_data_type ? data_type : input_data_type;

  auto primitive = op->primitive();
  auto storage_info_list =
    ops::RealImagViewBasicTypeCalc(primitive, {input_tensor}, is_real_view, is_complex_data_type);
  if (storage_info_list.empty()) {
    MS_EXCEPTION(ValueError) << "The storage info of RealView or ImagView input tensor is empty.";
  }
  tensor::TensorPtrList outputs;
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::CreateOutputTensor(op->device_context(), input_tensor, storage_info_list[0], &outputs, data_type);
  op->set_outputs(outputs);

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
    MS_LOG(DEBUG) << "View device task RealView start";
    auto device_context = op->device_context();
    PyBoostUtils::MallocOpInputsForView(device_context, input_tensor);
    MS_LOG(DEBUG) << "View device task RealView end";
  }));

  MS_LOG(DEBUG) << "View RealView Call end";
  return op->output(0);
}

tensor::TensorPtr RealViewAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  return RealImagViewAscendCustomizeBase(op, input_tensor, true);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
