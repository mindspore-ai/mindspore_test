/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_IR_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_IR_BUILDER_H_

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "utils/hash_map.h"
#include "common/common_utils.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/expander/base/node.h"
#include "backend/common/graph_kernel/expander/base/emitter.h"

namespace mindspore::graphkernel::expander {
class IrBuilder {
 public:
  IrBuilder() = default;
  virtual ~IrBuilder() = default;

  void Init(const EmitterPtr &emitter, const NodePtrList *inputs, const HashMap<std::string, ValuePtr> *attrs,
            const std::string &processor) {
    e = emitter;
    inputs_ptr_ = inputs;
    attrs_ptr_ = attrs;
    processor_ = processor;
  }
  virtual NodePtrList Expand() = 0;

  /// \brief build a Tensor node from shape
  NodePtr Tensor(std::vector<int64_t> input) const { return e->EmitValue(std::make_shared<tensor::Tensor>(input)); }

  /// \brief build a Tensor node from imm data
  template <typename T>
  NodePtr Tensor(T data, const TypePtr &type_ptr) const {
    return e->EmitValue(std::make_shared<tensor::Tensor>(data, type_ptr));
  }
  /// \brief build a Tensor node from data list
  NodePtr Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) const {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data_type, shape, data, src_data_type);
    return e->EmitValue(tensor_ptr);
  }
  /// \brief build a imm value node
  template <typename T>
  NodePtr Value(const T &value) const {
    return e->EmitValue(MakeValue(value));
  }

  const NodePtrList &inputs() const { return *inputs_ptr_; }
  const NodePtr &input(size_t i) const {
    if (i < inputs_ptr_->size()) {
      return (*inputs_ptr_)[i];
    }
    MS_LOG(EXCEPTION) << "Input index " << i << " out of range of input size " << inputs_ptr_->size();
  }
  const HashMap<std::string, ValuePtr> &attrs() const { return *attrs_ptr_; }
  ValuePtr attr(const std::string &key) const {
    auto iter = attrs_ptr_->find(key);
    return iter != attrs_ptr_->end() ? iter->second : nullptr;
  }
  template <typename T>
  T attr(const std::string &key) const {
    auto v = attr(key);
    MS_EXCEPTION_IF_NULL(v);
    return GetValue<T>(v);
  }
  NodePtr ScalarToTensor(const NodePtr &node, const TypePtr &dtype) const {
    MS_EXCEPTION_IF_NULL(node);
    auto value = node->GetValue();
    MS_EXCEPTION_IF_NULL(value);
    auto scalar = value->cast<ScalarPtr>();
    if (scalar == nullptr) {
      return Cast(node, dtype);
    }
    auto tensor = mindspore::ScalarToTensor(scalar, dtype);
    return e->EmitValue(tensor);
  }
  const std::string &processor() const { return processor_; }
  bool is_int_or_bool(const NodePtr &node) const {
    auto node_type = node->GetDtype()->type_id();
    return node_type == kNumberTypeInt32 || node_type == kNumberTypeBool;
  }
  // meta ops begin
  inline NodePtr Abs(const NodePtr &node) const { return e->Emit(MetaOp::Abs, {node}); }
  inline NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Add, {lhs, rhs}); }
  template <typename T>
  inline NodePtr Add(const NodePtr &x, T n) const {
    return Add(x, Tensor(n, x->GetDtype()));
  }
  inline NodePtr Assign(const NodePtr &dst, const NodePtr &src) const { return e->Emit(MetaOp::Assign, {dst, src}); }
  inline NodePtr BroadcastTo(const NodePtr &node, const NodePtr &shape) const {
    return e->Emit(MetaOp::BroadcastTo, {node, shape});
  }
  inline NodePtr Cast(const NodePtr &node, const NodePtr &dst_type) const {
    return e->Emit(MetaOp::Cast, {node, dst_type});
  }
  inline NodePtr Cast(const NodePtr &node, const TypePtr &dst_type) const { return Cast(node, dst_type->type_id()); }
  inline NodePtr Cast(const NodePtr &node, TypeId dst_type) const {
    return Cast(node, Value(static_cast<int64_t>(dst_type)));
  }
  inline NodePtr Floor(const NodePtr &node) const { return e->Emit(MetaOp::Floor, {node}); }
  inline NodePtr Trunc(const NodePtr &node) const { return e->Emit(MetaOp::Trunc, {node}); }
  inline NodePtr Concat(const NodePtrList &inputs, const NodePtr &axis) const {
    NodePtrList new_inputs(inputs.cbegin(), inputs.cend());
    new_inputs.push_back(axis);
    return e->Emit(MetaOp::Concat, new_inputs);
  }
  inline NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Div, {lhs, rhs}); }
  template <typename T>
  inline NodePtr Div(const NodePtr &x, T n) const {
    return Div(x, Tensor(n, x->GetDtype()));
  }
  inline NodePtr RealDiv(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::RealDiv, {lhs, rhs}); }
  template <typename T>
  inline NodePtr RealDiv(const NodePtr &x, T n) const {
    return RealDiv(x, Tensor(n, x->GetDtype()));
  }
  inline NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Equal, {lhs, rhs}); }
  inline NodePtr Exp(const NodePtr &node) const { return e->Emit(MetaOp::Exp, {node}); }
  inline NodePtr Gather(const NodePtr &param, const NodePtr &indices, const NodePtr &axis) const {
    return e->Emit(MetaOp::Gather, {param, indices, axis});
  }
  inline NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Greater, {lhs, rhs}); }
  inline NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs) const {
    return e->Emit(MetaOp::GreaterEqual, {lhs, rhs});
  }
  inline NodePtr IsInf(const NodePtr &node) const { return e->Emit(MetaOp::IsInf, {node}); }
  inline NodePtr IsNan(const NodePtr &node) const { return e->Emit(MetaOp::IsNan, {node}); }
  inline NodePtr Less(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Less, {lhs, rhs}); }
  inline NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs) const {
    return e->Emit(MetaOp::LessEqual, {lhs, rhs});
  }
  inline NodePtr Log(const NodePtr &node) const { return e->Emit(MetaOp::Log, {node}); }
  inline NodePtr LogicalAnd(const NodePtr &lhs, const NodePtr &rhs) const {
    return e->Emit(MetaOp::LogicalAnd, {lhs, rhs});
  }
  inline NodePtr LogicalOr(const NodePtr &lhs, const NodePtr &rhs) const {
    return e->Emit(MetaOp::LogicalOr, {lhs, rhs});
  }
  inline NodePtr LogicalNot(const NodePtr &node) const { return e->Emit(MetaOp::LogicalNot, {node}); }
  inline NodePtr MatMul(const NodePtr &a, const NodePtr &b, const NodePtr &transpose_a,
                        const NodePtr &transpose_b) const {
    return e->Emit(MetaOp::MatMul, {a, b, transpose_a, transpose_b});
  }
  inline NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Mul, {lhs, rhs}); }
  template <typename T>
  inline NodePtr Mul(const NodePtr &x, T n) const {
    return Mul(x, Tensor(n, x->GetDtype()));
  }
  inline NodePtr Neg(const NodePtr &node) const { return e->Emit(MetaOp::Neg, {node}); }
  inline NodePtr ReduceMax(const NodePtr &node, const NodePtr &axis, const NodePtr &keepdims) const {
    return e->Emit(MetaOp::ReduceMax, {node, axis, keepdims});
  }
  inline NodePtr ReduceMin(const NodePtr &node, const NodePtr &axis, const NodePtr &keepdims) const {
    return e->Emit(MetaOp::ReduceMin, {node, axis, keepdims});
  }
  inline NodePtr ReduceSum(const NodePtr &node, const NodePtr &axis, const NodePtr &keepdims) const {
    return e->Emit(MetaOp::ReduceSum, {node, axis, keepdims, Value(false)});
  }
  inline NodePtr ReduceSum(const NodePtr &node, const ShapeVector &axis, bool keepdims) const {
    return ReduceSum(node, Tensor(axis), Value(keepdims));
  }
  inline NodePtr Reshape(const NodePtr &node, const NodePtr &shape) const {
    return e->Emit(MetaOp::Reshape, {node, shape});
  }
  inline NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) const { return Reshape(node, Tensor(shape)); }
  inline NodePtr Rsqrt(const NodePtr &node) const { return e->Emit(MetaOp::Rsqrt, {node}); }
  inline NodePtr Reciprocal(const NodePtr &node) const { return e->Emit(MetaOp::Reciprocal, {node}); }
  inline NodePtr Select(const NodePtr &cond, const NodePtr &true_case, const NodePtr &false_case) const {
    return e->Emit(MetaOp::Select, {cond, true_case, false_case});
  }
  inline NodePtr Shape(const NodePtr &node) const { return e->Emit(MetaOp::Shape, {node}); }
  inline NodePtr Sqrt(const NodePtr &node) const { return e->Emit(MetaOp::Sqrt, {node}); }
  inline NodePtr StridedSlice(const NodePtr &input, const NodePtr &begin, const NodePtr &end,
                              const NodePtr &strides) const {
    return e->Emit(MetaOp::StridedSlice, {input, begin, end, strides});
  }
  inline NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Sub, {lhs, rhs}); }
  template <typename T>
  inline NodePtr Sub(const NodePtr &x, T n) const {
    return Sub(x, Tensor(n, x->GetDtype()));
  }
  inline NodePtr Tanh(const NodePtr &node) const {
    // Tanh(x) = 1 - 2/(e^{2x}+1)
    auto f32 = TypeIdToType(kNumberTypeFloat32);
    auto need_cast = node->GetDtype() != f32;
    auto cast_node = need_cast ? Cast(node, f32) : node;
    auto tanh_exp = Exp(Mul(cast_node, Tensor(2, f32)));
    auto tanh_add_0 = Add(tanh_exp, Tensor(1, f32));
    auto tanh_rec = Reciprocal(tanh_add_0);
    auto tanh_neg = Mul(tanh_rec, Tensor(-2, f32));
    auto tanh_add_1 = Add(tanh_neg, Tensor(1, f32));
    auto res = need_cast ? Cast(tanh_add_1, node->GetDtype()) : tanh_add_1;
    return res;
  }
  inline NodePtr Cosh(const NodePtr &node) const {
    // Cosh(x) = (e^x + e^{-x})/2
    auto f32 = TypeIdToType(kNumberTypeFloat32);
    auto need_cast = node->GetDtype() != f32;
    auto cast_node = need_cast ? Cast(node, f32) : node;
    auto cosh_exp_pos = Exp(cast_node);
    auto cosh_exp_neg = Exp(Mul(cast_node, Tensor(-1, f32)));
    auto cosh_add = Add(cosh_exp_pos, cosh_exp_neg);
    auto cosh_div = Div(cosh_add, Tensor(2, f32));
    auto res = need_cast ? Cast(cosh_div, node->GetDtype()) : cosh_div;
    return res;
  }
  inline NodePtr Sinh(const NodePtr &node) const {
    auto f32 = TypeIdToType(kNumberTypeFloat32);
    auto need_cast = node->GetDtype() != f32;
    auto cast_node = need_cast ? Cast(node, f32) : node;
    auto sinh_exp_pos = Exp(cast_node);
    auto sinh_exp_neg = Exp(Mul(cast_node, Tensor(-1, f32)));
    auto sinh_add = Sub(sinh_exp_pos, sinh_exp_neg);
    auto sinh_div = Div(sinh_add, Tensor(2, f32));
    auto res = need_cast ? Cast(sinh_div, node->GetDtype()) : sinh_div;
    return res;
  }
  inline NodePtr TensorScatterAdd(const NodePtr &input, const NodePtr &indices, const NodePtr &update) const {
    return e->Emit(MetaOp::TensorScatterAdd, {input, indices, update});
  }
  inline NodePtr Transpose(const NodePtr &node, const NodePtr &perm) const {
    return e->Emit(MetaOp::Transpose, {node, perm});
  }
  inline NodePtr Maximum(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Maximum, {lhs, rhs}); }
  inline NodePtr Minimum(const NodePtr &lhs, const NodePtr &rhs) const { return e->Emit(MetaOp::Minimum, {lhs, rhs}); }
  template <typename T>
  inline NodePtr Maximum(const NodePtr &lhs, T num) const {
    return Maximum(lhs, Tensor(num, lhs->GetDtype()));
  }
  template <typename T>
  inline NodePtr Minimum(const NodePtr &lhs, T num) const {
    return Minimum(lhs, Tensor(num, lhs->GetDtype()));
  }
  // meta ops end
 protected:
  EmitterPtr e;
  const NodePtrList *inputs_ptr_{nullptr};
  const HashMap<std::string, ValuePtr> *attrs_ptr_{nullptr};
  std::string processor_;
};

class DefaultIrBuilder : public IrBuilder {
 public:
  using ExpandFunc = std::function<NodePtrList(const DefaultIrBuilder *)>;
  explicit DefaultIrBuilder(const ExpandFunc &func, const std::string &name) : func_(func), name_(name) {}
  ~DefaultIrBuilder() override = default;

  NodePtrList Expand() override { return func_(this); }

  const EmitterPtr &emitter() const { return e; }
  const std::string &name() const { return name_; }

 protected:
  ExpandFunc func_;
  std::string name_;  // name of op
};

class IrBuilderRegistry {
 public:
  using CreatorFunc = std::function<std::unique_ptr<IrBuilder>()>;
  static IrBuilderRegistry &Instance() {
    static IrBuilderRegistry reg{};
    return reg;
  }
  class RegHelper {
   public:
    // Register IrBuilder by subclass.
    RegHelper(const std::string &name, const CreatorFunc &func) { IrBuilderRegistry::Instance().Reg(name, func); }
    // Register DefaultIrBuilder
    explicit RegHelper(const std::string &name) : name_(name) {}
    RegHelper &SetBody(const DefaultIrBuilder::ExpandFunc &func) {
      IrBuilderRegistry::Instance().Reg(
        name_, [func, name = this->name_]() { return std::make_unique<DefaultIrBuilder>(func, name); });
      return *this;
    }
    RegHelper &SetRealOutputIndices(const std::vector<size_t> &&output_indices) {
      IrBuilderRegistry::Instance().RegOutputIndices(name_, output_indices);
      return *this;
    }

    ~RegHelper() = default;

   protected:
    std::string name_;
    DefaultIrBuilder::ExpandFunc expand_func_;
  };

  bool HasOp(const std::string &name) const { return creator_map_.count(name) > 0; }
  bool IsOutputNumInconsistent(const std::string &name) { return output_num_inconsistent_ops_.count(name) > 0; }
  const HashMap<std::string, std::vector<size_t>> &GetOutputNumInconsistentOps() {
    return output_num_inconsistent_ops_;
  }
  std::unique_ptr<IrBuilder> GetOp(const std::string &name) const {
    auto iter = creator_map_.find(name);
    return (iter != creator_map_.end() ? iter->second() : nullptr);
  }

 private:
  IrBuilderRegistry() = default;
  ~IrBuilderRegistry() = default;

  void Reg(const std::string &name, const CreatorFunc &func) { creator_map_[name] = func; }
  void RegOutputIndices(const std::string &name, const std::vector<size_t> &output_indices) {
    output_num_inconsistent_ops_[name] = output_indices;
  }
  HashMap<std::string, CreatorFunc> creator_map_;
  // For some ops, the output number of the original cnode is different from the output number of subgraph.
  // The common reason is: the output of Assign must be an output of graph kernel,
  // but the parameter be assigned is unnecessary to be an output of the original cnode.
  // These ops are recorded in output_num_inconsistent_ops_
  HashMap<std::string, std::vector<size_t>> output_num_inconsistent_ops_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define BODYFUNC(v) [](const DefaultIrBuilder *v) -> NodePtrList

#define REG_EXPANDER_CLASS(name, cls)                                            \
  static const IrBuilderRegistry::RegHelper UNIQUE_NAME(g_op_cls_, __COUNTER__)( \
    name, []() noexcept { return std::unique_ptr<IrBuilder>(static_cast<IrBuilder *>(new cls())); })

#define REG_EXPANDER_FUNC(name) \
  static const IrBuilderRegistry::RegHelper UNIQUE_NAME(g_op_func, __COUNTER__) = IrBuilderRegistry::RegHelper(name)
}  // namespace mindspore::graphkernel::expander
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_EXPANDER_BASE_IR_BUILDER_H_
