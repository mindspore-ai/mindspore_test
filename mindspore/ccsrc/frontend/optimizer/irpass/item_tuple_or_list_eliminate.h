/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <map>

#include "frontend/optimizer/optimizer_caller.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
// (a, b, c, ...)[-1] => (a, b, c, ...)[length-1]
// [a, b, c, ...][-1] => [a, b, c, ...][length-1]
// (a, b, c, ...)[-1] = z => (a, b, c, ...)[length - 1] = z
// [a, b, c, ...][-1] = z => [a, b, c, ...][length - 1] = z
// {prim::kPrimTupleGetItem, T, N}
// {prim::kPrimListGetItem, L, N}
// {prim::kPrimTupleSetItem, T, N, Z}
// {prim::kPrimListSetItem, L, N, Z}
const std::map<std::string, size_t> kSliceAttrToIndex = {{kSliceStart, 1}, {kSliceStop, 2}, {kSliceStep, 3}};

class TupleListConvertItemIndexToPositive : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsCNode, IsVNode, IsNode})(node);

    FuncGraphPtr fg = node->func_graph();
    if (is_match_ && fg != nullptr) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      constexpr auto index_input = 2;
      inputs[index_input] = NewValueNode(id_);
      return fg->NewCNode(inputs);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override { sequeue_ = cnode; }

  void Visit(const ValueNodePtr &vnode) override {
    if (sequeue_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        if (sequeue_->abstract() == nullptr) {
          return;
        }
        auto sequeue_abstract = sequeue_->abstract()->cast<abstract::AbstractSequencePtr>();
        if (sequeue_abstract == nullptr || sequeue_abstract->dynamic_len()) {
          return;
        }
        id_ = idx + SizeToLong(sequeue_abstract->size());
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    sequeue_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  int64_t id_{0};
  CNodePtr sequeue_{nullptr};
};

// {prim::kPrimSliceGetItem, {prim::kPrimMakeSlice (a,b,c)}} => a
class MakeSliceSliceGetItemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimSliceGetItem, {IsCNode, IsVNode})(node);
    if (is_match_) {
      return make_slice_->input(idx_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeSlice)) {
      make_slice_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (make_slice_ != nullptr && IsValueNode<StringImm>(vnode)) {
      auto slice_attr_ = GetValue<std::string>(vnode->value());
      auto iter = kSliceAttrToIndex.find(slice_attr_);
      if (iter == kSliceAttrToIndex.end()) {
        MS_EXCEPTION_WITH_NODE(ValueError, vnode) << "The slice must be [start, stop, step], but got " << slice_attr_;
      }
      idx_ = iter->second;
      if (idx_ > make_slice_->size()) {
        MS_EXCEPTION_WITH_NODE(IndexError, vnode)
          << "The node make_slice should has 3 inputs but got " << make_slice_->DebugString();
      }
      is_match_ = true;
    }
  }

  void Reset() {
    idx_ = 0;
    make_slice_ = nullptr;
    is_match_ = false;
  }

 private:
  CNodePtr make_slice_ = nullptr;
  size_t idx_ = 0;
  bool is_match_{false};
};

// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, {prim::kPrimMakeTuple, Xs}, C}
// {prim::kPrimListGetItem, {prim::kPrimMakeList, Xs}, C}
class TupleListGetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);

    if (is_match_) {
      return tuple_->input(id_);
    }
    return nullptr;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
      tuple_ = cnode;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + SizeToLong(tuple_->size()) - 1;
      }
      id_ = LongToSize(idx + 1);
      if (tuple_->size() > id_) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    tuple_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  CNodePtr tuple_{nullptr};
};

// (a, b, c, ...)[0] => a
// (a, b, c, ...)[1] => b
// {prim::kPrimTupleGetItem, C1, C}
// {prim::kPrimListGetItem, C1, C}
class TupleListGetitemConstEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsVNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsVNode, IsVNode})(node);

    if (is_match_) {
      auto out = NewValueNode((*seq_)[id_]);
      out->set_has_new_value(has_new_value_);
      out->set_abstract((*seq_)[id_]->ToAbstract());
      return out;
    }
    return nullptr;
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<ValueSequence>(vnode)) {
      seq_ = GetValueNode<ValueSequencePtr>(vnode);
      has_new_value_ = vnode->has_new_value();
    }
    if (seq_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + SizeToLong(seq_->size());
      }
      id_ = LongToSize(idx);
      if (id_ < seq_->size()) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    seq_ = nullptr;
    is_match_ = false;
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  ValueSequencePtr seq_{nullptr};
  bool has_new_value_{false};
};

// (a, b, c, ...)[0] = z => (z, b, c, ...)
// [a, b, c, ...][0] = z => [z, b, c, ...]
// {prim::kPrimTupleSetItem, {prim::kPrimMakeTuple, a, b, c, ...}, 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, {prim::kPrimMakeList, a, b, c, ...}, 0, z} => {prim::kPrimMakeList, z, b, c, ...}
// {prim::kPrimTupleSetItem, (a, b, c, ...), 0, z} => {prim::kPrimMakeTuple, z, b, c, ...}
// {prim::kPrimListSetItem, [a, b, c, ...], 0, z} => {prim::kPrimMakeList, z, b, c, ...}
class TupleListSetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsCNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimTupleSetItem, {IsVNode, IsVNode, IsNode})(node);
    AnfVisitor::Match(prim::kPrimListSetItem, {IsVNode, IsVNode, IsNode})(node);

    auto fg = node->func_graph();
    if (fg != nullptr && z_ != nullptr) {
      args_[id_] = z_;
      return fg->NewCNode(args_);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_match_) {
      z_ = node;
      return;
    }

    AnfVisitor::Visit(node);
  }

  void Visit(const CNodePtr &cnode) override {
    CNodePtr real_node = cnode;
    while (IsPrimitiveCNode(real_node, prim::kPrimDepend)) {
      auto depend = real_node->cast<CNodePtr>();
      real_node = depend->input(1)->cast<CNodePtr>();
    }
    if (IsPrimitiveCNode(real_node, prim::kPrimMakeTuple) || IsPrimitiveCNode(real_node, prim::kPrimMakeList)) {
      auto &inputs = real_node->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(args_));
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (args_.empty() && IsValueNode<ValueTuple>(vnode)) {
      auto tuple = GetValueNode<ValueTuplePtr>(vnode);
      if (tuple != nullptr) {
        (void)args_.emplace_back(NewValueNode(prim::kPrimMakeTuple));
        for (auto &val : tuple->value()) {
          auto val_node = std::make_shared<ValueNode>(val);
          val_node->set_abstract(val->ToAbstract());
          (void)args_.emplace_back(val_node);
        }
      }
    } else if (args_.empty() && IsValueNode<ValueList>(vnode)) {
      auto list = GetValueNode<ValueListPtr>(vnode);
      if (list != nullptr) {
        (void)args_.emplace_back(NewValueNode(prim::kPrimMakeList));
        for (auto &val : list->value()) {
          auto val_node = std::make_shared<ValueNode>(val);
          val_node->set_abstract(val->ToAbstract());
          (void)args_.emplace_back(val_node);
        }
      }
    } else if (!args_.empty() && IsValueNode<Int64Imm>(vnode)) {
      auto idx = GetValue<int64_t>(vnode->value());
      if (idx < 0) {
        idx = idx + SizeToLong(args_.size()) - 1;
      }
      id_ = LongToSize(idx + 1);
      if (id_ < args_.size()) {
        is_match_ = true;
      }
    }
  }

  void Reset() {
    id_ = 0;
    z_ = nullptr;
    is_match_ = false;
    args_.clear();
  }

 private:
  bool is_match_{false};
  size_t id_{0};
  AnfNodePtr z_{nullptr};
  std::vector<AnfNodePtr> args_{};
};

// {prim::kPrimTupleGetItem, {prim::kPrimTupleSetItem, Y, C1, X}, C2}
// {prim::kPrimListGetItem, {prim::kPrimListSetItem, Y, C1, X}, C2}
class TupleListGetSetitemEliminator : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsVNode})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsVNode})(node);

    if (set_item_index_ < 0 || get_item_index_ < 0) {
      return nullptr;
    }
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    if (set_item_index_ == get_item_index_) {
      return set_item_value_;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto new_tuple_getitem =
      fg->NewCNode({cnode->input(kAnfPrimitiveIndex), set_item_tuple_, get_item_index_value_node_});
    new_tuple_getitem->set_abstract(node->abstract());
    return new_tuple_getitem;
  }

  void Visit(const CNodePtr &cnode) override {
    if (IsPrimitiveCNode(cnode, prim::kPrimTupleSetItem) || IsPrimitiveCNode(cnode, prim::kPrimListSetItem)) {
      if (cnode->size() < kTupleSetItemInputSize) {
        return;
      }
      set_item_tuple_ = cnode->input(kTupleSetItemTupleIndex);
      set_item_index_ = GetPositiveIndex(cnode->input(kTupleSetItemIndexIndex));
      set_item_value_ = cnode->input(kTupleSetItemValueIndex);
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (set_item_tuple_ != nullptr && IsValueNode<Int64Imm>(vnode)) {
      get_item_index_ = GetPositiveIndex(vnode);
      get_item_index_value_node_ = vnode;
    }
  }

  void Reset() {
    set_item_index_ = -1;
    get_item_index_ = -1;
    get_item_index_value_node_ = nullptr;
    set_item_value_ = nullptr;
    set_item_tuple_ = nullptr;
  }

 private:
  // TupleSetItem: {primTupleSetItem, set_item_tuple_, ValueNode{set_item_index_}, set_item_value_}
  int64_t set_item_index_{-1}, get_item_index_{-1};
  AnfNodePtr set_item_tuple_{nullptr}, set_item_value_{nullptr}, get_item_index_value_node_{nullptr};

  int64_t GetPositiveIndex(const AnfNodePtr &node) {
    auto vnode = node->cast<ValueNodePtr>();
    if (vnode == nullptr) {
      return -1;
    }
    auto value = vnode->value();
    MS_EXCEPTION_IF_NULL(value);
    auto int64_imm_value = value->cast_ptr<Int64Imm>();
    // If index is AnyValue or other not int64 type value, should not optimize the tuple_getitem.
    if (int64_imm_value == nullptr) {
      return -1;
    }
    auto index = int64_imm_value->value();
    if (index < 0) {
      MS_EXCEPTION_IF_NULL(set_item_tuple_->abstract());
      auto sequence_abstract = set_item_tuple_->abstract()->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abstract);
      if (sequence_abstract->dynamic_len()) {
        return -1;
      }
      index = index + SizeToLong(sequence_abstract->size());
    }
    if (index < 0) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Unexpected negative index:" << index << " , node: " << node->DebugString();
    }
    return index;
  }
};

// {prim::kPrimTupleGetItem, {prim::kPrimDepend, X, Y}, C} => {prim::kPrimDepend, {prim::kPrimTupleGetItem, X, C}, Y}
// {prim::kPrimListGetItem, {prim::kPrimDepend, X, Y}, C} => {prim::kPrimDepend, {prim::kPrimListGetItem, X, C}, Y}
class TupleListGetitemDependReorder : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &opt, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTupleGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    AnfVisitor::Match(prim::kPrimListGetItem, {IsCNode, IsValueNode<Int64Imm>})(node);
    if (x_ == nullptr) {
      return nullptr;
    }
    auto mgr = opt->manager();
    MS_EXCEPTION_IF_NULL(mgr);
    auto depend = node->cast<CNodePtr>()->input(1);
    auto depend_cnode = depend->cast<CNodePtr>();
    auto fg = node->func_graph();
    if (fg != depend_cnode->func_graph()) {
      return nullptr;
    }
    // Avoid generating redundant depend nodes.
    if (ExistUpdateStateUser(mgr, depend)) {
      auto inputs = node->cast<CNodePtr>()->inputs();
      inputs[1] = depend_cnode->input(1);
      auto new_node = fg->NewCNode(inputs);
      new_node->set_abstract(node->abstract());
      return new_node;
    }

    AnfNodePtr item_node;
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      item_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), x_, c_}, fg);
    } else {
      item_node = NewCNode({NewValueNode(prim::kPrimListGetItem), x_, c_}, fg);
    }
    item_node->set_abstract(node->abstract());
    auto depend_node = NewCNode({depend_cnode->input(0), item_node, y_}, fg);
    depend_node->set_abstract(node->abstract());
    auto abs = x_->abstract();
    if (abs == nullptr) {
      return depend_node;
    }
    auto idx_value = GetValueNode<Int64ImmPtr>(c_);
    MS_EXCEPTION_IF_NULL(idx_value);
    int64_t idx = idx_value->value();
    if (abs->isa<abstract::AbstractSequence>()) {
      auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
      if (idx < 0) {
        idx += SizeToLong(abs_seq->elements().size());
      }
      if (idx < 0 || LongToSize(idx) >= abs_seq->elements().size()) {
        MS_LOG_WITH_NODE(EXCEPTION, c_) << "The idx value " << idx << " of tuple_getitem node " << c_->DebugString()
                                        << " is out of range.";
      }
      item_node->set_abstract(abs_seq->elements()[LongToSize(idx)]);
    }
    depend_node->set_abstract(item_node->abstract());
    return depend_node;
  }

  bool ExistUpdateStateUser(const FuncGraphManagerPtr &mgr, const AnfNodePtr &depend) {
    if (!IsPrimitiveCNode(y_, prim::kPrimUpdateState)) {
      return false;
    }
    auto &node_users = mgr->node_users();
    auto iter = node_users.find(depend);
    if (iter == node_users.end()) {
      return false;
    }
    auto &users = iter->second;
    if (users.size() <= 1) {
      return false;
    }
    bool has_updatestate_user = std::any_of(users.begin(), users.end(), [](const auto &user) {
      return IsPrimitiveCNode(user.first, prim::kPrimUpdateState);
    });
    return has_updatestate_user;
  }

  void Visit(const CNodePtr &cnode) override {
    // {prim::kPrimDepend, X, Y}
    constexpr auto depend_input_size = 3;
    constexpr auto depend_index_first = 1;
    constexpr auto depend_index_second = 2;
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->size() == depend_input_size) {
      x_ = cnode->input(depend_index_first);
      y_ = cnode->input(depend_index_second);
    }
  }

  void Visit(const ValueNodePtr &vnode) override { c_ = vnode; }

  void Reset() {
    x_ = nullptr;
    y_ = nullptr;
    c_ = nullptr;
  }

 private:
  AnfNodePtr x_{nullptr}, y_{nullptr}, c_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ITEM_TUPLE_OR_LIST_ELIMINATE_H_
