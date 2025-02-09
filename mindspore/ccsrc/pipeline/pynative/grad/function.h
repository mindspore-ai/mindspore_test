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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <type_traits>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"
#include "pipeline/pynative/grad/variable.h"

namespace mindspore::pynative::autograd {

using BaseTensorPtrList = std::vector<BaseTensorPtr>;

inline BaseTensorPtrList ToTensorList(const BaseTensorPtr &tensor) { return BaseTensorPtrList{tensor}; }

inline BaseTensorPtrList ToTensorList(const BaseTensorPtrList &tensor_list) { return tensor_list; }

template <typename T>
std::enable_if_t<std::is_same_v<T, BaseTensorPtrList>, T> ToOutputType(const BaseTensorPtrList &tensor_list) {
  return tensor_list;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, BaseTensorPtr>, T> ToOutputType(const BaseTensorPtrList &tensor_list) {
  return tensor_list[0];
}

inline std::string GetFunctionTypeName(const char *name) {
#ifdef _MSC_VER
  return name;
#else
  int status = -1;
  std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name;
#endif
}

void PrepareForForward();

template <typename X, typename... Args>
using forward_t = decltype(X::Forward(nullptr, std::declval<Args>()...));

template <class T>
struct Function {
  template <typename X = T, typename... Args>
  static auto Apply(Args &&... args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>>;
};

struct CppFunctionContext {
  CppFunctionContext() = default;

  void SaveForBackward(BaseTensorPtrList to_save) { to_save_ = std::move(to_save); }
  BaseTensorPtrList GetSavedTensors() { return to_save_; }
  void MarkDirty(const BaseTensorPtrList &inputs);
  void MarkNonDifferentiable(const BaseTensorPtrList &outputs);
  void SetMaterializeGrads(bool value) { materialize_grads_ = value; }
  // for backward
  bool NeedsInputGrad(size_t output_edge_index) const;
  // for forward
  bool NeedGrad(const BaseTensorPtr &tensor);

  std::unordered_map<std::string, ValuePtr> saved_data;

  // NOLINTNEXTLINE(runtime/references)
  friend void CppFunctionDoGrad(CppFunctionContext *context, BaseTensorPtrList &inputs, BaseTensorPtrList &outputs);

 private:
  BaseTensorPtrList to_save_;
  std::unordered_set<BaseTensorPtr> non_differentiable_;
  std::unordered_set<BaseTensorPtr> dirty_inputs_;
  bool materialize_grads_{true};
  std::weak_ptr<BackwardNode> node_;

  template <class T>
  friend struct CppFunctionNode;
};

template <class T>
struct CppFunctionNode : public BackwardNode {
  explicit CppFunctionNode(string name, size_t output_size = 1) : BackwardNode(std::move(name), output_size) {}
  ~CppFunctionNode() override = default;
  ValuePtrList CallBackward(const ValuePtrList &grads) override;
  void Release() override;
  void SetContextBackwardNode(const BackwardNodePtr &node) { context_.node_ = node; }
  void SetOutputSize(size_t output_size) { output_size_ = output_size; }
  CppFunctionContext context_;
  std::vector<bool> is_tensor_input_;
  abstract::AbstractBasePtrList outputs_abstract_;
};

template <class T>
template <typename X, typename... Args>
auto Function<T>::Apply(Args &&... args) -> std::enable_if_t<std::is_same_v<X, T>, forward_t<X, Args...>> {
  // create CppFunctionNode
  auto function_name = std::string("Function[") + GetFunctionTypeName(typeid(T).name()) + "]";
  MS_LOG(DEBUG) << function_name << " Begin Apply";
  auto node_ptr = std::make_shared<CppFunctionNode<T>>(function_name);
  // process function input
  BaseTensorPtrList input_vars;
  auto check_is_base_tensor = [&input_vars, &node_ptr](auto &&arg) {
    using ArgType = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<ArgType, BaseTensorPtr>) {
      arg->set_need_pipeline_sync(true);
      (void)input_vars.emplace_back(arg);
      node_ptr->is_tensor_input_.emplace_back(true);
    } else {
      node_ptr->is_tensor_input_.emplace_back(false);
    }
  };
  (check_is_base_tensor(std::forward<Args>(args)), ...);

  // prepare for forward
  PrepareForForward();

  // forward
  MS_LOG(DEBUG) << function_name << " Begin Forward";
  using forward_return_t = forward_t<X, Args...>;
  forward_return_t output;
  {
    kernel::pyboost::RequireGradGuard require_grad_guard(false);
    output = T::Forward(&(node_ptr->context_), std::forward<Args>(args)...);
  }

  BaseTensorPtrList output_list = ToTensorList(output);
  MS_EXCEPTION_IF_CHECK_FAIL(output_list.size() > 0, "The output list must not be empty.");
  AbstractBasePtrList outputs_abstract;
  outputs_abstract.reserve(output_list.size());
  for (size_t i = 0; i < output_list.size(); i++) {
    AbstractBasePtr abs = output_list[i]->ToAbstract();
    abs->set_value(kValueAny);
    (void)outputs_abstract.emplace_back(abs);
  }

  node_ptr->SetContextBackwardNode(node_ptr);
  node_ptr->outputs_abstract_ = outputs_abstract;
  node_ptr->SetOutputSize(output_list.size());

  // set autograd
  CppFunctionDoGrad(&(node_ptr->context_), input_vars, output_list);
  return ToOutputType<forward_return_t>(output_list);
}

BaseTensorPtrList GradPreProcess(const ValuePtrList &grads, const AbstractBasePtrList &outputs_abstract,
                                 bool materialize_grads, const std::string &function_name);

ValuePtrList GradPostProcess(const BaseTensorPtrList &outputs, std::vector<bool> is_tensor_input,
                             const std::string &function_name);

template <class T>
ValuePtrList CppFunctionNode<T>::CallBackward(const ValuePtrList &grads) {
  auto grad_in = GradPreProcess(grads, outputs_abstract_, context_.materialize_grads_, name_);
  auto grad_out = T::Backward(&context_, grad_in);
  return GradPostProcess(grad_out, is_tensor_input_, name_);
}

template <class T>
void CppFunctionNode<T>::Release() {
  context_.to_save_.clear();
  context_.saved_data.clear();
}

}  // namespace mindspore::pynative::autograd

namespace pybind11::detail {
template <>
struct type_caster<mindspore::tensor::BaseTensorPtr> {
  PYBIND11_TYPE_CASTER(mindspore::tensor::BaseTensorPtr, _("Tensor"));
  bool load(handle src, bool);
  static handle cast(const mindspore::tensor::BaseTensorPtr &src, return_value_policy, handle);
};
}  // namespace pybind11::detail
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
