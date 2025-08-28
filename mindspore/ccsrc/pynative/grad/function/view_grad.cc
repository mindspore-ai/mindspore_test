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

#include "mindspore/ccsrc/pynative/grad/function/view_grad.h"
#include <memory>
#include "mindspore/core/include/utils/device_manager_conf.h"
#include "pyboost/functions/auto_generate/functions.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"

namespace mindspore::pynative::autograd {
namespace {
inline void SetDeviceTarget() {
  const auto &device_target = DeviceManagerConf::GetInstance()->device_type();
  kernel::pyboost::OpRunStatus::Get().set_run_info(kernel::pyboost::OpStatus(true, false, 0, device_target));
}
}  // namespace

ValuePtrList ViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto output = kernel::pyboost::reshape(grads[0]->cast<TensorPtr>(), self_shape_);
  return {output};
}

ValuePtrList TransposeBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto ndims = perm_.size();
  std::vector<int64_t> invert_perm(ndims);
  for (size_t i = 0; i < ndims; ++i) {
    invert_perm[ops::DynamicDimWrap(perm_[i], static_cast<int64_t>(ndims))] = i;
  }
  auto output = kernel::pyboost::transpose(grads[0]->cast<TensorPtr>(), invert_perm);
  return {output};
}

ValuePtrList TransposeExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto output = kernel::pyboost::transpose_ext_view(grads[0]->cast<TensorPtr>(), dim0_, dim1_);
  return {output};
}

ValuePtrList SelectExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto size = PackBasicTypeToValue(self_shape_);
  auto grad = grads[0]->cast<TensorPtr>();
  auto dtype = std::make_shared<Int64Imm>(grad->data_type());
  auto grad_input = kernel::pyboost::zeros(size, dtype);
  auto select_part = kernel::pyboost::select_ext_view(grad_input, dim_, index_);
  (void)kernel::pyboost::inplace_copy(select_part, grad, std::make_shared<BoolImm>(true));
  return {grad_input};
}

ValuePtrList SliceExtViewBackwardNode::CallBackward(const ValuePtrList &grads) {
  SetDeviceTarget();
  auto size = PackBasicTypeToValue(self_shape_);
  auto grad = grads[0]->cast<TensorPtr>();
  auto dtype = std::make_shared<Int64Imm>(grad->data_type());
  auto grad_input = kernel::pyboost::zeros(size, dtype);
  auto slice_part = kernel::pyboost::slice_ext_view(grad_input, dim_, start_, end_, step_);
  (void)kernel::pyboost::inplace_copy(slice_part, grad, std::make_shared<BoolImm>(true));
  return {grad_input};
}
}  // namespace mindspore::pynative::autograd
