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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "ir/anf.h"
#include "ir/meta_grad_data.h"
#include "ir/base_tensor.h"
#include "include/common/utils/hook.h"

namespace mindspore {
namespace session {
class KernelGraph;
}
using KernelGraphPtr = std::shared_ptr<session::KernelGraph>;
}  // namespace mindspore

namespace mindspore::pynative::autograd {
class FuncBuilder;
struct GradAttr {
  GradAttr(bool get_all, bool get_by_list, bool sens_param, bool get_by_position, bool weight_param_is_tuple)
      : grad_all_inputs(get_all),
        grad_weights(get_by_list),
        has_sens(sens_param),
        get_by_position(get_by_position),
        weight_param_is_tuple(weight_param_is_tuple) {}

  bool grad_all_inputs;
  bool grad_weights;
  bool has_sens;
  bool get_by_position;
  bool weight_param_is_tuple;
};

class AutoGradMetaData : public AutoGradMetaInterface {
 public:
  AutoGradMetaData() = default;
  AutoGradMetaData(size_t op_index, const InputType input_type) : op_index_(op_index), input_type_(input_type) {}
  AutoGradMetaData(const VariablePtr &variable, const ParameterPtr &parameter,
                   const InputType input_type = InputType::kConstant)
      : variable_(variable), parameter_(parameter), input_type_(input_type) {}
  [[nodiscard]] VariablePtr UnsafeGetVariableImpl() const override { return variable_.lock(); }
  void set_variable(const VariablePtr &variable) override { variable_ = variable; }
  [[nodiscard]] ParameterPtr parameter() const override { return parameter_.lock(); }
  void set_parameter(const ParameterPtr &parameter) override { parameter_ = parameter; }
  void set_k_node(const AnfNodePtr &k_node) override { k_node_ = k_node; }
  [[nodiscard]] AnfNodePtr k_node() const override { return k_node_.lock(); }
  [[nodiscard]] InputType input_type() const override { return input_type_; }
  void set_input_type(InputType input_type) override { input_type_ = input_type; }
  [[nodiscard]] size_t op_index() const override { return op_index_; }
  void set_op_index(size_t op_index) override { op_index_ = op_index; }
  [[nodiscard]] size_t output_index() const override { return output_index_; }
  void set_output_index(size_t output_index) override { output_index_ = output_index; }
  // Reset Parameter auto grad meta
  void Reset() override {
    variable_ = {};
    parameter_ = {};
    k_node_ = {};
    op_index_ = 0;
    output_index_ = 0;
    input_type_ = InputType::kUnkown;
  }
  void AddBackwardHook(uint64_t id, TensorBackwardHookPtr hook) {
    (void)backward_hooks_.emplace(id, std::move(hook));
    is_register_hook_ = true;
  }
  void RemoveBackwardHook(uint64_t id) { (void)backward_hooks_.erase(id); }
  bool is_register_hook() const override { return is_register_hook_; }
  const std::map<uint64_t, TensorBackwardHookPtr> &backward_hooks() override { return backward_hooks_; }
  void ClearBackwardHooks() { backward_hooks_.clear(); }
  ~AutoGradMetaData() override = default;

 private:
  // Weakptr for variable, to avoid circular reference
  VariableWeakPtr variable_;
  // Weakptr to hold ir parameter of input or parameter
  ParameterWeakPtr parameter_;
  // Weakptr to k_node for tensor
  AnfNodeWeakPtr k_node_;
  // Optional for op output, represent index of op in execute order.
  size_t op_index_{0};
  // Type of grad tensor
  InputType input_type_{InputType::kUnkown};
  // Index of op output tensors.
  size_t output_index_{0};
  bool is_register_hook_{false};
  // Tensor hooks
  std::map<uint64_t, TensorBackwardHookPtr> backward_hooks_{};
};
using AutoGradMetaDataPtr = std::shared_ptr<AutoGradMetaData>;

using BaseTensor = tensor::BaseTensor;
using BaseTensorPtr = std::shared_ptr<tensor::BaseTensor>;

class ViewInfo {
 public:
  explicit ViewInfo(BaseTensorPtr base) : base_(std::move(base)) {}
  [[nodiscard]] ViewInfo Union() const { return ViewInfo(base_); }
  [[nodiscard]] const tensor::BaseTensorPtr &base() const { return base_; }

 private:
  BaseTensorPtr base_;
};

enum class CreationType {
  // View created in grad mode.
  kDefault = 0,
  // View created in no grad mode.
  kNoGradMode,
  // View created by multi output op.
  kMultiOutput,
  // View created by custom bprop.
  kCustomBprop,
};

class ViewAutoGradMetaData final : public AutoGradMetaData {
 public:
  ViewAutoGradMetaData(const ViewInfo &&view_info, size_t op_index, InputType input_type,
                       CreationType creation_type = CreationType::kDefault)
      : AutoGradMetaData(op_index, input_type), view_info_(view_info), creation_type_(creation_type) {}
  [[nodiscard]] const ViewInfo &view_info() const { return view_info_; }
  [[nodiscard]] uint32_t version_attr() const { return version_attr_; }
  void set_version_attr(uint32_t version) { version_attr_ = version; }
  CreationType creation_type() { return creation_type_; }
  void set_creation_type(const CreationType &creation_type) { creation_type_ = creation_type; }

 private:
  ViewInfo view_info_;
  CreationType creation_type_;
  // We need set version attr in bprop queue to avoid multi thread race.
  uint32_t version_attr_{0};
};
using ViewAutoGradMetaDataPtr = std::shared_ptr<ViewAutoGradMetaData>;

class Variable;
struct Edge {
  /// \brief Constructor.
  ///
  /// \param[in] variable The variable represents object need gradient.
  /// \param[in] input_index The input index is variable output index.
  explicit Edge(std::shared_ptr<Variable> variable, size_t input_index)
      : variable(std::move(variable)), input_index(input_index) {}
  // Just a placeholder.
  Edge() : variable(nullptr), input_index(0) {}
  // Check edge is defined, if is defined, it mean that this edge is effective.
  // We need use undefined edge as placeholder, so that we can known operator input index exactly,
  // for example, when we use copy operator, we will knonw it has two tensor input, and next_edges[0] is self tensor.
  // so that when we use inplace op, we can skip self's edge and update other edges.
  [[nodiscard]] inline bool is_defined() const { return variable != nullptr; }
  std::shared_ptr<Variable> variable;
  size_t input_index;
};

class PYNATIVE_EXPORT BackwardNode {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] name The name represents op name.
  /// \param[in] output_size The output_size is output size for op.
  explicit BackwardNode(std::string name, size_t output_size = 1) : name_(std::move(name)), output_size_(output_size) {}

  /// \brief Destructor.
  virtual ~BackwardNode() = default;

  /// \brief CallBackward function is used to calculate gradient of this node.
  ///
  /// \param[in] grads Grads is this node output's gradients.
  virtual ValuePtrList CallBackward(const ValuePtrList &grads) { return {}; }

  /// \brief Postprocess gradients from func to align with next_edges.
  /// \param[in] gradient_value Gradients value is gradients result from func
  /// which need postprocess.
  /// \return Real gradients after postprocess, the size is same as next edges size.
  virtual ValuePtrList PostProcess(const ValuePtrList &gradient_value);

  // Update nullptr grad.
  ValuePtrList LazeUpdateZeroGradient(const ValuePtrList &dout, FuncBuilder *func_builder, const ValuePtr &output);

  /// \brief The PostProcess function is used to represent this node's inputs, which can
  /// backpropagation gradients.
  ///
  /// \return next edges
  const std::vector<Edge> &next_edges() const { return next_edges_; }

  /// \brief Set next edges for backward node.
  ///
  void set_next_edges(std::vector<Edge> &&next_edges) { next_edges_ = next_edges; }

  /// \brief Add next edges for backward node.
  ///
  void add_next_edge(Edge edge) { (void)next_edges_.emplace_back(std::move(edge)); }

  /// \brief name of this Node.
  ///
  /// \return name
  const std::string &name() { return name_; }

  /// \brief Check func to check whether the version of input is changed.
  ///
  /// \return check_func
  const std::function<void(const std::string &op_name)> &check_func() const { return check_func_; }

  /// \brief Set check func.
  ///
  void set_check_func(const std::function<void(const std::string &op_name)> &check_func) { check_func_ = check_func; }
  /// \brief Set op output value
  ///
  /// \return op output value
  void set_op_output(const ValuePtr &op_output) { op_output_ = op_output; }

  /// \brief Get op output value
  ///
  /// \return op output value
  const ValuePtr &op_output() { return op_output_; }

  /// \brief The size of node output.
  ///
  /// \return output size
  size_t output_size() const { return output_size_; }

  /// \brief Release resource
  ///
  /// \return void
  virtual void Release() {}

 protected:
  std::vector<Edge> next_edges_;
  std::string name_;
  std::function<void(const std::string &op_name)> check_func_{nullptr};
  ValuePtr op_output_{nullptr};
  size_t output_size_;
};
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

class IrFunctionNode {
 public:
  IrFunctionNode(KernelGraphPtr tape, const AnfNodePtr &dout)
      : tape_(std::move(tape)), accumulate_dout_(dout), fake_dout_(dout) {}
  void AddNextEdge(const std::shared_ptr<Variable> &next_variable, const AnfNodePtr &din);
  void UpdateAccumulativeDout(const AnfNodePtr &new_dout);
  [[nodiscard]] const std::vector<std::pair<std::shared_ptr<Variable>, AnfNodePtr>> &next_edges() const {
    return next_edges_;
  }
  const KernelGraphPtr &tape() { return tape_; }
  [[nodiscard]] const AnfNodePtr &accumulate_dout() const { return accumulate_dout_; }
  void set_accumulate_dout(const AnfNodePtr &accumulate_dout) { accumulate_dout_ = accumulate_dout; }
  void ReplaceEdges();
  [[nodiscard]] const AnfNodePtr &fake_dout() const { return fake_dout_; }

 private:
  AnfNodePtr HyperAdd(const AnfNodePtr &left_node, const AnfNodePtr &right_node);
  // Bprop func graph
  const KernelGraphPtr tape_;
  // Input of dout for this bprop function
  AnfNodePtr accumulate_dout_;
  // First we generate a fake dout
  const AnfNodePtr fake_dout_;
  // The pair.first is a variable, pair.second is dout of variable
  std::vector<std::pair<std::shared_ptr<Variable>, AnfNodePtr>> next_edges_;
  // Replace next_edges where din == dout in brprop function
  std::vector<int> need_replace_edges_;
};
using IrFunctionNodePtr = std::shared_ptr<IrFunctionNode>;

// Variable represent a tensor need grad
class Variable {
 public:
  /// \brief Constructor.
  ///
  Variable() = default;

  /// \brief Destructor.
  ///
  virtual ~Variable() = default;

  /// \param fn, Backward function.
  /// \param is_leaf, The variable is leaf or not.
  Variable(BackwardNodePtr &&fn, bool is_leaf) : is_leaf_(is_leaf), func_node_(std::move(fn)) {}

  /// \brief Constructor.
  ///
  /// \param fn, IrFunctionNodePtr function.
  /// \param is_leaf, The variable is leaf or not.
  Variable(IrFunctionNodePtr &&fn, ValuePtr &&out_value, bool is_leaf)
      : is_leaf_(is_leaf), out_value_(std::move(out_value)), ir_function_node_(std::move(fn)) {}

  /// \brief Backward function.
  ///
  /// \return fn
  BackwardNodePtr func_node() const { return func_node_; }

  /// \brief IrFunctionNode function.
  ///
  /// \return fn for ir
  IrFunctionNodePtr ir_function_node() const { return ir_function_node_; }

  /// \brief Name of fake op.
  ///
  /// \return fake_prim_name
  const string &fake_prim_name() const { return fake_prim_name_; }

  /// \brief Set name of fake op.
  ///
  /// \param fake_prim_name
  void set_fake_prim_name(const string &fake_prim_name) { fake_prim_name_ = fake_prim_name; }

  /// \brief Flag to judge whether the op is fake op.
  ///
  bool is_fake_bprop() const { return is_fake_bprop_; }

  /// \brief Set fake bprop.
  ///
  /// \param is_fake_bprop
  void set_is_fake_bprop(bool is_fake_bprop) { is_fake_bprop_ = is_fake_bprop; }

  /// \brief Flag to judge whether the op is custom bprop op.
  ///
  bool is_custom_op_variable() const { return is_custom_op_variable_; }

  /// \brief Set custom bprop flag.
  ///
  /// \param is_custom_op_variable
  void set_is_custom_op_variable(bool is_custom_op_variable) { is_custom_op_variable_ = is_custom_op_variable; }

  /// \brief Flag to judge whether the variable is need propagate.
  ///
  /// \return True if the variable need propagate, false if not.
  bool is_need_propagate() const { return is_need_propagate_; }

  /// \brief Set need propagate.
  ///
  void set_is_need_propagate(bool is_need_grad) { is_need_propagate_ = is_need_grad; }

  /// \brief Flag to judge whether the variable is need grad.
  ///
  /// \return is need grad
  bool is_need_grad() const { return is_need_grad_; }

  /// \brief Set need grad.
  ///
  /// \param is_need_grad
  void set_is_need_grad(bool is_need_grad) { is_need_grad_ = is_need_grad; }

  /// \brief Judge whether the variable is left node.
  ///
  /// \return True if variable is leaf, false if not.
  bool is_leaf() const { return is_leaf_; }

  /// \brief Get forward output value.
  ///
  /// \return valueptr.

  ValuePtr out_value() const { return out_value_; }

  /// \brief Debug info.
  ///
  /// \return debug info.
  virtual std::string ToString() const { return {}; }
  /// \brief Release input and output tensors
  ///
  /// \return void
  void Release();

 private:
  // If node has not bprop, we record its prim name
  std::string fake_prim_name_;
  // Record this node is a fake bprop
  bool is_fake_bprop_{false};
  // Flag to judge need to propagrate
  bool is_need_propagate_{false};
  // Flag to judge variable whether need grad
  bool is_need_grad_{true};
  // Variable is a custom bprop define
  bool is_custom_op_variable_{false};
  // Flag the variable is a leaf in bprop.
  bool is_leaf_{false};
  ValuePtr out_value_{nullptr};
  // Abstract bprop function
  BackwardNodePtr func_node_{nullptr};
  // Abstract bprop function
  IrFunctionNodePtr ir_function_node_{nullptr};
};
using VariablePtr = std::shared_ptr<Variable>;

// FuncVariable represent a parameter or output of op
class FuncVariable : public Variable {
 public:
  /// \brief Constructor.
  ///
  FuncVariable() = default;
  ~FuncVariable() override = default;
  /// \brief Constructor.
  ///
  /// \param fn, Backward function.
  /// \param is_leaf, The variable is leaf or not.
  explicit FuncVariable(BackwardNodePtr fn, bool is_leaf = false) : Variable(std::move(fn), is_leaf) {}

  /// \brief Gradients of the variable if variable is left node, nullptr if not left node.
  ///
  const tensor::BaseTensorPtr &grad() const { return grad_; }

  /// \brief Set gradients of the leaf variable.
  ///
  /// \param grad
  void set_grad(const tensor::BaseTensorPtr &grad) { grad_ = grad; }

  std::string ToString() const override;

 private:
  // Grad for this variable, only leaf node has grad.
  tensor::BaseTensorPtr grad_;
};
using FuncVariablePtr = std::shared_ptr<FuncVariable>;

// IrVariable represent a parameter or output of a middle cnode
class IrVariable : public Variable {
 public:
  IrVariable() = default;
  ~IrVariable() override = default;

  IrVariable(IrFunctionNodePtr fn, ValuePtr out_value, bool is_leaf = false)
      : Variable(std::move(fn), std::move(out_value), is_leaf) {}

  AnfNodePtr k_node() const { return k_node_; }
  void set_k_node(const AnfNodePtr &k_node) { k_node_ = k_node; }
  AnfNodePtr RealDout();
  std::string ToString() const override;

 private:
  AnfNodePtr k_node_{nullptr};
};
using IrVariablePtr = std::shared_ptr<IrVariable>;

template <typename T>
bool isa(const BackwardNodePtr &base_ptr) {
  const auto &object = (*base_ptr);
  return typeid(object) == typeid(T);
}

namespace impl {
AutoGradMetaDataPtr get_autograd_meta_impl(const tensor::BaseTensorPtr &tensor);
AutoGradMetaDataPtr get_autograd_meta_impl(const tensor::BaseTensor &tensor);
ViewAutoGradMetaDataPtr get_view_autograd_meta_impl(const tensor::BaseTensorPtr &tensor);
}  // namespace impl
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_VARIABLE_H_
