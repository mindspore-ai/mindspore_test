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

#include "pynative/grad/variable.h"
#include <memory>
#include "pynative/grad/grad_utils.h"
#include "pynative/pynative_utils.h"
#include "include/common/pynative/common_utils.h"

namespace mindspore::pynative::autograd {
ValuePtrList BackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  auto flatten_gradients = CommonUtils::FlattenTensorSeqInValueSeq(gradient_value, false);
  return flatten_gradients;
}

ValuePtrList BackwardNode::LazeUpdateZeroGradient(const ValuePtrList &dout, FuncBuilder *func_builder,
                                                  const ValuePtr &output) {
  if (dout.size() == kSizeOne) {
    return dout;
  }
  ValuePtrList outputs;
  CommonUtils::FlattenValueSeqArg(output, false, false, &outputs);
  if (outputs.size() != dout.size()) {
    MS_LOG(EXCEPTION) << "Gradients size should be same as output size! But got output size: " << outputs.size()
                      << ", gradients size: " << dout.size();
  }
  ValuePtrList real_dout(dout.size());
  for (size_t i = 0; i < dout.size(); ++i) {
    if (dout[i]->isa<None>()) {
      MS_LOG(DEBUG) << "Op " << name() << " has multi outputs, and exist null dout, now do emit zeros";
      auto zero_value = PyNativeAlgo::AutoGradUtil::BuildSpecialValueGrad(outputs[i], nullptr, func_builder,
                                                                          SpecialType::kZerosLikeType);
      MS_EXCEPTION_IF_NULL(zero_value);
      real_dout[i] = zero_value;
    } else {
      real_dout[i] = dout[i];
    }
  }
  return real_dout;
}

std::string FuncVariable::ToString() const {
  std::ostringstream buf;
  buf << "Variable name: " << func_node()->name() << ", is_need_grad: " << is_need_grad()
      << ", is_need_propagate: " << is_need_propagate() << " is_leaf: " << is_leaf() << "\n";
  for (size_t i = 0; i < func_node()->next_edges().size(); ++i) {
    if (!func_node()->next_edges()[i].is_defined()) {
      buf << "Last edge: " << i << " undefined edge"
          << "\n";
      continue;
    }
    auto last_variable = func_node()->next_edges()[i].variable;
    auto index = func_node()->next_edges()[i].input_index;
    MS_EXCEPTION_IF_NULL(last_variable->func_node());
    buf << "Last edge: " << i << ", variable name: " << last_variable->func_node()->name()
        << ", output index: " << index << "\n";
  }
  return buf.str();
}

std::string IrVariable::ToString() const {
  std::ostringstream buf;
  buf << "Variable id: " << PyNativeAlgo::Common::GetIdByValue(out_value());
  if (auto tensor = out_value()->cast<tensor::BaseTensorPtr>(); tensor != nullptr && tensor->is_parameter()) {
    buf << ", parameter name: " + tensor->param_info()->name();
  }
  buf << ", is_need_grad: " << is_need_grad() << ", is_need_propagate: " << is_need_propagate()
      << ", is_leaf: " << is_leaf();
  for (size_t i = 0; i < ir_function_node()->next_edges().size(); ++i) {
    auto last_variable = ir_function_node()->next_edges()[i].first;
    auto din = ir_function_node()->next_edges()[i].second;
    buf << ", next edge variable id: " << PyNativeAlgo::Common::GetIdByValue(last_variable->out_value())
        << " din: " << din->DebugString();
  }
  return buf.str();
}

AnfNodePtr IrVariable::RealDout() {
  if (static_cast<bool>(
        MS_UNLIKELY(PyNativeAlgo::AutoGradUtil::IsZerosLikeNode(ir_function_node()->accumulate_dout())))) {
    ir_function_node()->set_accumulate_dout(PyNativeAlgo::AutoGradUtil::BuildSpecialNode(
      ir_function_node()->tape(), out_value(), ir_function_node()->accumulate_dout()->abstract(),
      SpecialType::kZerosLikeType));
  }
  const auto &accumulate_dout = ir_function_node()->accumulate_dout();
  const auto &dout_abs = accumulate_dout->abstract();
  MS_EXCEPTION_IF_NULL(dout_abs);
  // For input, if it is a sparsetensor, we need return a sparsetensor.
  if (out_value()->isa<tensor::BaseTensor>() || dout_abs->isa<abstract::AbstractSparseTensor>()) {
    return accumulate_dout;
  }
  if (out_value()->isa<tensor::MetaSparseTensor>()) {
    return PyNativeAlgo::AutoGradUtil::BuildSparseTensorNode(ir_function_node()->tape(), out_value(), accumulate_dout);
  }
  return accumulate_dout;
}

namespace impl {
AutoGradMetaDataPtr get_autograd_meta_impl(const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  return get_autograd_meta_impl(*tensor);
}

AutoGradMetaDataPtr get_autograd_meta_impl(const tensor::BaseTensor &tensor) {
  auto auto_grad_meta = tensor.auto_grad_meta_data();
  if (auto_grad_meta == nullptr) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<AutoGradMetaData>(auto_grad_meta);
}

ViewAutoGradMetaDataPtr get_view_autograd_meta_impl(const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->auto_grad_meta_data() == nullptr) {
    return nullptr;
  }
  const auto &meta_data = tensor->auto_grad_meta_data();
  auto view_meta_data = std::dynamic_pointer_cast<ViewAutoGradMetaData>(meta_data);
  return view_meta_data;
}
}  // namespace impl
}  // namespace mindspore::pynative::autograd
