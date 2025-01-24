/*
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_

#include <string>
#include <map>
#include <set>
#include <stack>
#include <vector>
#include <memory>
#include <unordered_map>
#include "ops/op_def.h"
#include "frontend/operator/meta_dsl/common/utils.h"
#include "frontend/operator/meta_dsl/common/meta_func_builder.h"

namespace mindspore::prim {
using NodePtr = AnfNodePtr;
using NodePtrList = AnfNodePtrList;
using BlockFunc = std::function<void()>;
using CheckFunc = std::function<void(const PrimitivePtr &, const std::vector<AbstractBasePtr> &)>;

class MetaImpl : public MetaFuncGraph {
 public:
  explicit MetaImpl(const std::string &name) : MetaFuncGraph(name + "MetaImpl"), name_(name) {}
  ~MetaImpl() override = default;
  MS_DECLARE_PARENT(MetaImpl, MetaFuncGraph)
  void set_prim(const PrimitivePtr &prim);
  PrimitivePtr prim() const;
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &input_args) override;
  virtual void GenerateFunction() = 0;

 protected:
  /// \brief Get Primitive.
  ///
  /// \note Example: Prim(Add), Prim(Mul)
  ///
  /// \param[in] name The name of Primitive.
  ///
  /// \return Primitive instance.
#define Prim(name) kPrim##name

  /// \brief Create a node with value.
  ///
  /// \note Example: Value(0), Value(1.0), Value(true), Value("valid"), Value<int32_t>(100), Value(kNone)
  ///
  /// \param[in] value Supports int, float, bool, char*,  and other types allowed by MakeValue.
  ///
  /// \return ValueNode.
  template <typename S, typename U = typename ImmTraits<S>::type::element_type>
  inline ValueNodePtr Value(S value) {
    if (std::is_same_v<S, int>) {
      // int defaults to Int64.
      return NewValueNode(std::make_shared<Int64Imm>(static_cast<int64_t>(value)));
    }
    return NewValueNode(std::make_shared<U>(value));
  }
  inline ValueNodePtr Value(const ValuePtr &value) { return NewValueNode(value); }
  inline ValueNodePtr Value(const std::vector<ValuePtr> &v) { return NewValueNode(std::make_shared<ValueTuple>(v)); }
  inline ValueNodePtr Value(std::initializer_list<ValuePtr> v) { return NewValueNode(std::make_shared<ValueTuple>(v)); }

  /// \brief Create a call node, whose first input is usually a Primitive.
  ///
  /// \note Example: Call(Prim(Add), x, y), Call(Prim(Rank), x)
  ///
  /// \param[in] prim A primitive.
  /// \param[in] args Nodes as inputs.
  ///
  /// \return CNode.
  template <typename... TArgs>
  inline NodePtr Call(const PrimitivePtr &prim, const TArgs &... args) {
    NodePtr prim_node = nullptr;
    if (ops::IsPrimitiveFunction(prim->name())) {
      prim_node = NewValueNode(std::make_shared<prim::DoTransPrimitiveFunction>(prim));
    } else {
      prim_node = NewValueNode(prim);
    }
    NodePtrList nodes = {prim_node, args...};
    return NewNode(nodes);
  }
  template <typename... TArgs>
  inline NodePtr Call(const TArgs &... args) {
    NodePtrList nodes = {args...};
    return NewNode(nodes);
  }

  /// \brief Set output.
  ///
  /// \note Example: Return(out)
  ///
  /// \param[in] output The output of graph.
  void Return(const NodePtr &output);

  /// \brief if-else expression.
  ///
  /// \note Example:
  ///         # python                       // cpp
  ///         if condition:                  auto true_case = [&]() { Return(x); };
  ///           return x         -->         auto false_case = [&]() { Return(y); };
  ///         return y                       auto out = If(condition, true_case, false_case, (x, y))
  ///
  /// \param[in] cond The condition of if-else expression.
  /// \param[in] true_case True branch.
  /// \param[in] false_case False branch.
  /// \param[in] params All parameters used by true branch and false branch.
  ///
  /// \return The result node of if-else expression.
#define If(cond, true_case, false_case, params) IF_IMPL(cond, true_case, false_case, params)

  /// \brief Create a new tuple, such as (x, y).
  ///
  /// \note Example: Tuple(x, y)
  ///
  /// \param[in] args Input nodes.
  ///
  /// \return Node with MakeTuple.
  template <typename... TArgs>
  inline NodePtr Tuple(const TArgs &... args) {
    return Call(NewValueNode(prim::kPrimMakeTuple), args...);
  }

  /// \brief Create a new list, such as [0, 1, 2].
  ///
  /// \note Example: List(NewValue(0), NewValue(1), NewValue(2))
  ///
  /// \param[in] args Input nodes.
  ///
  /// \return Node with MakeList.
  template <typename... TArgs>
  inline NodePtr List(const TArgs &... args) {
    return Call(NewValueNode(prim::kPrimMakeList), args...);
  }

  /// \brief x == y
  ///
  /// \note Example: Equal(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Equal.
  NodePtr Equal(const NodePtr &x, const NodePtr &y);

  /// \brief x != y
  ///
  /// \note Example: NotEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of NotEqual.
  NodePtr NotEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x > y
  ///
  /// \note Example: Greater(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Greater.
  NodePtr Greater(const NodePtr &x, const NodePtr &y);

  /// \brief x < y
  ///
  /// \note Example: Less(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Less.
  NodePtr Less(const NodePtr &x, const NodePtr &y);

  /// \brief x >= y
  ///
  /// \note Example: GreaterEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of GreaterEqual.
  NodePtr GreaterEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x <= y
  ///
  /// \note Example: LessEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of LessEqual.
  NodePtr LessEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x[y]
  ///
  /// \note Example: GetItem(x, y)
  ///
  /// \param[in] x The object used with getitem.
  /// \param[in] y Index.
  ///
  /// \return Output node of GetItem.
  NodePtr GetItem(const NodePtr &x, const NodePtr &y);

  /// \brief x[y] = z
  ///
  /// \note Example: SetItem(x, y, z)
  ///
  /// \param[in] x The object used with setitem.
  /// \param[in] y Index.
  /// \param[in] z New element.
  ///
  /// \return Output node of SetItem.
  NodePtr SetItem(const NodePtr &x, const NodePtr &y, const NodePtr &z);

  /// \brief x is None
  ///
  /// \note Example: IsNone(x)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Output node, used to determine whether node is None type.
  NodePtr IsNone(const NodePtr &node);

  /// \brief x is not None
  ///
  /// \note Example: IsNotNone(x)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Output node, used to determine whether node is None type.
  NodePtr IsNotNone(const NodePtr &node);

  /// \brief x and y
  ///
  /// \note Example: And(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr And(const NodePtr &x, const NodePtr &y);

  /// \brief x or y
  ///
  /// \note Example: Or(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr Or(const NodePtr &x, const NodePtr &y);

  /// \brief x + y
  ///
  /// \note Example: ScalarAdd(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarAdd(const NodePtr &x, const NodePtr &y);

  /// \brief x - y
  ///
  /// \note Example: ScalarSub(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarSub(const NodePtr &x, const NodePtr &y);

  /// \brief x * y
  ///
  /// \note Example: ScalarMul(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarMul(const NodePtr &x, const NodePtr &y);

  /// \brief x // y
  ///
  /// \note Example: ScalarFloorDiv(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarFloorDiv(const NodePtr &x, const NodePtr &y);

  /// \brief x % y
  ///
  /// \note Example: ScalarMod(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarMod(const NodePtr &x, const NodePtr &y);

  /// \brief x ** y
  ///
  /// \note Example: ScalarPow(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarPow(const NodePtr &x, const NodePtr &y);

  /// \brief x's shape
  ///
  /// \note Example: Shape(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Shape(const NodePtr &x);

  /// \brief x's rank
  ///
  /// \note Example: Rank(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Rank(const NodePtr &x);

  /// \brief reshape x
  ///
  /// \note Example: Reshape(x, shape)
  ///
  /// \param[in] x Input node.
  /// \param[in] shape Input node.
  ///
  /// \return Output node.
  NodePtr Reshape(const NodePtr &x, const NodePtr &shape);

  /// \brief not x
  ///
  /// \note Example: Not(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Not(const NodePtr &x);

  /// \brief Raise exception such as ValueError and TypeError.
  ///
  /// \note Example: Raise("ValueError", "Not supported yet")
  ///
  /// \param[in] exception_type Exception type
  /// \param[in] exception_msg Exception log message.
  ///
  /// \return Node with prim::kPrimRaise.
  NodePtr Raise(const std::string &exception_type, const std::string &exception_msg);

  // Tools for implementing macro definitions, and they are basically not used during development.
  NodePtr NewParam(const std::string &name);
  NodePtr IfCond(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch,
                 const NodePtrList &args);
  void set_check_func(const CheckFunc &check_func);

 private:
  void BeginFunc(size_t params_size, const std::string &func_name = "anonymous");
  FuncGraphPtr EndFunc();
  NodePtr NewNode(const NodePtrList &nodes);
  void CheckInputs(const AbstractBasePtrList &input_args) const;
  FuncGraphPtr BuildSubFunction(const std::string &func_name, const BlockFunc &sub_func, size_t n_args);
  void DumpIRForMetaDsl(const FuncGraphPtr &graph) const;

  PrimitivePtr prim_{nullptr};
  std::string name_;
  CheckFunc check_func_{nullptr};
  std::stack<MetaFuncBuilderPtr> func_builder_stack_;
};
using MetaImplPtr = std::shared_ptr<MetaImpl>;
using CreateFunc = std::function<std::shared_ptr<MetaImpl>()>;

bool IsMetaImpl(const std::string &name);
void AddMetaImpl(const std::string &name, const CreateFunc &creator);
MetaImplPtr CreateMetaImpl(const std::string &name);

class MetaImplRegHelper {
 public:
  MetaImplRegHelper(const std::string &name, const CreateFunc &creator) { AddMetaImpl(name, creator); }
  ~MetaImplRegHelper() = default;
};

#define REGISTER_FUNCTION_OP(name, check_func)                                  \
  class name##MetaImpl : public MetaImpl {                                      \
   public:                                                                      \
    explicit name##MetaImpl() : MetaImpl(#name) { set_check_func(check_func); } \
    ~name##MetaImpl() override = default;                                       \
    MS_DECLARE_PARENT(name##MetaImpl, MetaImpl)                                 \
    void GenerateFunction() override;                                           \
  };                                                                            \
  static const MetaImplRegHelper meta_impl_helper_##name(#name, []() { return std::make_shared<name##MetaImpl>(); });
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_
