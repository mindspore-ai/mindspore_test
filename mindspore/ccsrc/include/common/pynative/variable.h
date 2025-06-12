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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <unordered_set>
#include "ir/anf.h"
#include "ir/meta_grad_data.h"
#include "ir/tensor.h"
#include "pynative/grad/hook_py.h"

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

class COMMON_EXPORT SavedNode {
 public:
  SavedNode() = default;
  SavedNode(ValuePtr data, std::shared_ptr<BackwardNode> grad_node, bool is_view_inplace, bool is_placeholder)
      : data_(std::move(data)),
        weak_grad_node_(grad_node),
        is_view_inplace_(is_view_inplace),
        is_placeholder_(is_placeholder) {}
  ValuePtr Unwrap(BackwardNodePtr grad_node, bool only_tensor = false);
  static std::shared_ptr<SavedNode> ConstructSavedNode(const ValuePtr &output, bool is_view_inplace = false);

 private:
  ValuePtr data_{nullptr};
  // Because when view inplace happen, inplace output grad node is not it's self,
  // so we need add weak reference for view output.
  std::weak_ptr<BackwardNode> weak_grad_node_{};
  bool is_view_inplace_{false};
  bool is_placeholder_{false};
};
using SavedNodePtr = std::shared_ptr<SavedNode>;

class TensorDescriptor {
 public:
  TensorDescriptor() = default;
  TensorDescriptor(std::vector<int64_t> shape, std::vector<int64_t> strides, TypePtr dtype, size_t storage_offset)
      : shape_(std::move(shape)),
        strides_(std::move(strides)),
        dtype_(std::move(dtype)),
        storage_offset_(storage_offset) {}

  [[nodiscard]] const std::vector<int64_t> &shape() const { return shape_; }
  [[nodiscard]] const std::vector<int64_t> &strides() const { return strides_; }
  [[nodiscard]] const TypePtr &dtype() const { return dtype_; }
  [[nodiscard]] size_t storage_offset() const { return storage_offset_; }

 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  TypePtr dtype_;
  size_t storage_offset_{0};
};

class BackwardNode;
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

class COMMON_EXPORT AutoGradMetaData : public AutoGradMetaInterface {
 public:
  AutoGradMetaData() = default;
  explicit AutoGradMetaData(const InputType input_type) : input_type_(input_type) {}
  explicit AutoGradMetaData(BackwardNodePtr grad_node, const InputType input_type = InputType::kConstant)
      : grad_node_(std::move(grad_node)), input_type_(input_type) {}
  [[nodiscard]] BackwardNodePtr UnsafeGetGradNodeImpl() const override { return grad_node_; }
  void set_grad_node(const BackwardNodePtr &grad_node) override { grad_node_ = grad_node; }
  [[nodiscard]] InputType input_type() const override { return input_type_; }
  void set_input_type(InputType input_type) override { input_type_ = input_type; }
  [[nodiscard]] size_t output_index() const override { return output_index_; }
  void set_output_index(size_t output_index) override { output_index_ = output_index; }
  // Reset Parameter auto grad meta
  void Reset() override {
    grad_node_ = nullptr;
    output_index_ = 0;
    input_type_ = InputType::kUnkown;
  }
  ~AutoGradMetaData() override = default;

 private:
  // grad_node for call grad fn.
  BackwardNodePtr grad_node_;
  // Type of grad tensor
  InputType input_type_{InputType::kUnkown};
  // Index of op output tensors.
  size_t output_index_{0};
};

using AutoGradMetaDataPtr = std::shared_ptr<AutoGradMetaData>;
using Tensor = tensor::Tensor;
using TensorPtr = std::shared_ptr<tensor::Tensor>;
using TensorPtrSet = std::unordered_set<tensor::TensorPtr>;

class ViewInfo {
 public:
  explicit ViewInfo(TensorPtr base) : base_(std::move(base)) {}
  [[nodiscard]] ViewInfo Union() const { return ViewInfo(base_); }
  [[nodiscard]] const tensor::TensorPtr &base() const { return base_; }

 private:
  TensorPtr base_;
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
  ViewAutoGradMetaData(const ViewInfo &&view_info, InputType input_type,
                       CreationType creation_type = CreationType::kDefault)
      : AutoGradMetaData(input_type), view_info_(view_info), creation_type_(creation_type) {}
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

struct Edge {
  /// \brief Constructor.
  /// \param[in] Grad node the grad node represents object need calculate gradient.
  /// \param[in] input_index The input index is variable output index.
  explicit Edge(BackwardNodePtr grad_node, size_t input_index)
      : grad_node(std::move(grad_node)), input_index(input_index) {}
  // Just a placeholder.
  Edge() : grad_node(nullptr), input_index(0) {}
  // Check edge is defined, if is defined, it mean that this edge is effective.
  // We need use undefined edge as placeholder, so that we can known operator input index exactly,
  // for example, when we use copy operator, we will knonw it has two tensor input, and next_edges[0] is self tensor.
  // so that when we use inplace op, we can skip self's edge and update other edges.
  [[nodiscard]] inline bool is_defined() const { return grad_node != nullptr; }
  BackwardNodePtr grad_node;
  size_t input_index;
};

class COMMON_EXPORT BackwardNode : public std::enable_shared_from_this<BackwardNode> {
 public:
  /// \brief Constructor.
  /// \param name
  /// \param output_size
  explicit BackwardNode(string name, size_t output_size = 1) noexcept;
  /// \brief Constructor.
  /// \param[in] name The name represents op name.
  /// \param[in] output_size The output_size is output size for op.
  explicit BackwardNode(string name, uint64_t seq_id, size_t output_size) noexcept;
  /// \brief Destructor.
  virtual ~BackwardNode() = default;
  DISABLE_COPY_AND_ASSIGN(BackwardNode);

  /// \brief CallBackward function is used to calculate gradient of this node.
  /// \param[in] grads Grads is this node output's gradients.
  virtual ValuePtrList CallBackward(const ValuePtrList &grads) { return {}; }

  /// \brief Postprocess gradients from func to align with next_edges.
  /// \param[in] gradient_value Gradients value is gradients result from func
  /// which need postprocess.
  /// \return Real gradients after postprocess, the size is same as next edges size.
  virtual ValuePtrList PostProcess(const ValuePtrList &gradient_value);

  /// \brief Add python tensor hook.
  /// \param id
  /// \param hook
  void AddPyTensorHook(uint64_t id, std::unique_ptr<PyTensorBackwardNodePreHook> &&hook) {
    py_tensor_pre_hooks_[id] = std::move(hook);
  }

  /// \brief Remove python tensor hook.
  /// \param id
  void RemovePyTensorHook(uint64_t id) { (void)py_tensor_pre_hooks_.erase(id); }

  /// check next edges is all not defined.
  /// \return true
  bool IsEmpty();

  /// \brief The PostProcess function is used to represent this node's inputs, which can
  /// backpropagation gradients.
  /// \return next edges
  const std::vector<Edge> &next_edges() const { return next_edges_; }

  /// \brief Set next edge for backward node.
  void set_next_edge(Edge &&edge, size_t i) { next_edges_[i] = std::move(edge); }

  /// \brief Set next edges for backward node.
  void set_next_edges(std::vector<Edge> &&next_edges) { next_edges_ = next_edges; }

  /// \brief Add next edges for backward node.
  void add_next_edge(Edge edge) { (void)next_edges_.emplace_back(std::move(edge)); }

  /// \brief name of this Node.
  /// \return name
  const std::string &name() const { return name_; }

  /// \brief Check func to check whether the version of input is changed.
  /// \return check_func
  const std::function<void(const std::string &op_name)> &check_func() const { return check_func_; }

  /// \brief Set check func.
  void set_check_func(const std::function<void(const std::string &op_name)> &check_func) { check_func_ = check_func; }

  /// \brief Backward hook for backward node.
  /// \return backward_hooks
  const OrderedMap<uint64_t, std::unique_ptr<PyTensorBackwardNodePreHook>> &py_tensor_pre_hooks() const {
    return py_tensor_pre_hooks_;
  }

  /// \brief The sequence number of current node.
  /// \return sequence number
  size_t seq_id() const { return seq_id_; }

  /// \brief The size of node output.
  /// \return output size
  size_t output_size() const { return output_size_; }

  /// \brief Release resource
  /// \return void
  virtual void Release() {}

  /// \brief Generate description str of node.
  /// \return string
  std::string ToString() const;

 protected:
  std::vector<Edge> next_edges_;
  std::string name_;
  std::function<void(const std::string &op_name)> check_func_{nullptr};
  // Tensor hooks
  OrderedMap<uint64_t, std::unique_ptr<PyTensorBackwardNodePreHook>> py_tensor_pre_hooks_;
  size_t seq_id_;
  size_t output_size_;
};
using BackwardNodePtr = std::shared_ptr<BackwardNode>;

template <typename T>
bool isa(const BackwardNodePtr &base_ptr) {
  const auto &object = (*base_ptr);
  return typeid(object) == typeid(T);
}

template <typename T>
bool isa(const BackwardNode *base_ptr) {
  const auto &object = (*base_ptr);
  return typeid(object) == typeid(T);
}

class COMMON_EXPORT AutoDiffInterface {
 public:
  [[nodiscard]] virtual bool IsInExecGraph(const BackwardNodePtr &node) const = 0;
  virtual void AddNodeToExecGraph(const BackwardNodePtr &node) = 0;
};
using AutoDiffInterfacePtr = std::shared_ptr<AutoDiffInterface>;

class COMMON_EXPORT AutoDiffGuard {
 public:
  explicit AutoDiffGuard(const AutoDiffInterfacePtr &auto_diff);
  ~AutoDiffGuard();

 private:
  AutoDiffInterfacePtr prev_auto_diff_engine_;
};

namespace impl {
COMMON_EXPORT AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT AutoGradMetaDataPtr GetAutogradMetaImpl(const tensor::Tensor &tensor);
COMMON_EXPORT ViewAutoGradMetaDataPtr GetViewAutogradMetaImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT BackwardNodePtr GetUnsafeGradNodeImpl(const tensor::TensorPtr &tensor);
COMMON_EXPORT bool RequiresGrad(const tensor::TensorPtr &tensor);
COMMON_EXPORT AutoDiffInterfacePtr CurrentAutoDiffEngine();
}  // namespace impl
}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_VARIABLE_H_
