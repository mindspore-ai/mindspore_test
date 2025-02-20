/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pynative/base.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "pynative/forward/forward_task.h"
#include "pynative/grad/function/func_builder.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pynative/grad/variable.h"
#include "pynative/grad/auto_grad.h"

namespace mindspore {
namespace pynative {
using CallBackFn = std::function<VectorRef(const VectorRef &arg_list)>;
enum class SpecialType { kZerosLikeType = 0, kOnesLikeType = 1 };

class BpropCallback final : public expander::bprop::PynativeCallback {
 public:
  BpropCallback(const PrimitivePtr &prim, ValuePtrList *inputs, ValuePtr *output)
      : prim_(prim), inputs_(inputs), output_(output) {}
  const std::string &opname() const override { return prim_->name(); }
  ValuePtr *GetInput(size_t index) const override { return &(*inputs_)[index]; }
  ValuePtrList *GetInputs() const override { return inputs_; }
  ValuePtr *GetOutput() const override { return output_; }
  bool IsNotRequiresGrad(size_t index) const override;
  void FreeDeviceAddress(ValuePtr *value) const override;

 private:
  const PrimitivePtr &prim_;
  ValuePtrList *inputs_;
  ValuePtr *output_;
};

namespace PyNativeAlgo {
struct AutoGradUtil {
  // Common grad function
  static InputType SetValueGradInfo(const ValuePtr &value, InputType grad_type);
  static InputType SetTensorGradInfo(const tensor::BaseTensorPtr &tensor);
  static ValuePtr BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence, size_t op_index = 0);
  static ValuePtr VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence,
                                   size_t op_index = 0);
  static void BuildViewAutoGradMeta(const tensor::BaseTensorPtr &src_tensor, const tensor::BaseTensorPtr &output,
                                    size_t op_index, autograd::CreationType creation_type);
  static void SetInferOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op);
  static void SetInferMultiOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op);
  static ValuePtr MakeOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                             const tensor::BaseTensorPtr &base_view = nullptr);
  static ValuePtr MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                                  const tensor::BaseTensorPtr &view_base = nullptr);
  // Multi inputs and multi outputs view op enter here, temp code need discard.
  static ValuePtr MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                                  const ValueTuplePtr &base_view);
  static void BumpVersion(const ValuePtr &value);

  static bool IsPrimNeedGrad(const PrimitivePtr &prim);
  static bool NeedGrad(const tensor::BaseTensorPtr &input_tensor);
  static bool NeedGrad(const std::vector<ValuePtr> &input_values);
  static bool IsZerosLikeNode(const AnfNodePtr &node);
  static ValuePtr GetFakeZeroTensor();
  static ValuePtr BuildSpecialValueGrad(const ValuePtr &value, const tensor::BaseTensorPtr &grad,
                                        autograd::FuncBuilder *func_builder, const SpecialType &type);
  static AnfNodePtr BuildSpecialNode(const KernelGraphPtr &tape, const ValuePtr &value,
                                     const abstract::AbstractBasePtr &abs, const SpecialType &type);
  static AnfNodePtr BuildSparseTensorNode(const KernelGraphPtr &tape, const ValuePtr &sparse_value,
                                          const AnfNodePtr &dout_value_node);
  static void SetGradMetaData(const ValuePtr &value, const VariablePtr &variable, const ParameterPtr &param = nullptr);
  static void SetGradInfoForInputs(const ValuePtr &value, const VariablePtr &variable,
                                   autograd::MetaGradInfoList *param_meta_grad_info,
                                   const ParameterPtr &param = nullptr);
  static inline bool IsParam(InputType grad_type) {
    return grad_type == InputType::kParameter || grad_type == InputType::kInput;
  }
  static inline bool IsParamRequiresGrad(const tensor::BaseTensorPtr &tensor) {
    return tensor->param_info() != nullptr && tensor->param_info()->requires_grad();
  }
  // Create fake bprop
  static void BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs);
  static CallBackFn CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                        const GraphCallCondition &graph_call_condition);
  static PrimitivePyPtr BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute = false);
  static void CheckRecomputeInputs(const ValuePtrList &inputs, bool is_need_recompute);
  static void ClearAutoGradStaticCache();
  static void CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info);
  static void CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs);
  static void CheckAndCloneInplaceInput(const kernel::pyboost::OpPtr &inplace_op, const PrimitivePtr &prim,
                                        const std::string &device_target, ValuePtrList &&inputs);
};
};  // namespace PyNativeAlgo
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_GRAD_GRAD_UTILS_H_
