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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_AUTO_GRAD_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_AUTO_GRAD_H_

#include <utility>
#include <string>
#include <memory>
#include <vector>

#include "ir/anf.h"
#include "pynative/base.h"
#include "pynative/grad/ir/ir_bprop.h"
#include "pynative/grad/custom_function.h"
#include "pynative/grad/function_py.h"

namespace mindspore::pynative::autograd {
using MetaGradInfoList = OrderedMap<tensor::BaseTensorPtr, AutoGradMetaDataPtr>;

class AutoGrad {
 public:
  AutoGrad() = default;
  virtual ~AutoGrad() = default;

  inline IrBprop *ir_bprop() const {
    MS_EXCEPTION_IF_NULL(ir_bprop_);
    return ir_bprop_.get();
  }
  // Used for high grad and jit
  virtual bool KPynativeWithFProp(const GradParamPtr &grad_param) { return true; }

  // Used for single op
  virtual bool KPynativeOp(const GradParamPtr &grad_param) { return true; }

  // Update top cell output, record last_node
  virtual void UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {}

  // Call custom bprop procedure
  virtual void CallCustomBprop(const CustomContext &context) {}

  // Call custom function procedure
  virtual void CallCustomFunction(const std::shared_ptr<FunctionContext> &context) {}

  // Call cpp function bprop
  virtual void CallCPPFunctionBprop(const ValuePtrList &flatten_outputs, const BaseTensorPtrSet &input_base_tensors,
                                    const BaseTensorPtrSet &dirty_tensors, const BaseTensorPtrSet &non_diff_tensors,
                                    const ValuePtrList &inputs, const std::vector<InputType> &input_value_grad_type,
                                    const BackwardNodePtr &node) {}

  virtual VariablePtr SafeGetVariableImpl(const tensor::BaseTensorPtr &tensor) { return nullptr; }

  // Store grad meta grad info
  MetaGradInfoList &param_meta_grad_info() { return param_meta_grad_info_; }

 protected:
  bool grad_by_value_{true};
  std::string device_target_;
  IrBpropPtr ir_bprop_;
  MetaGradInfoList param_meta_grad_info_;
};
using AutoGradPtr = std::shared_ptr<AutoGrad>;
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_AUTO_GRAD_H_
