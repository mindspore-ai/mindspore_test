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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_

#include <map>
#include <memory>

#include "ir/func_graph.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/comparison_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "ir/pattern_matcher.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/ps/parse/resolve.h"

namespace mindspore {
namespace opt {
namespace irpass {
// LoopUnroll for ops.Scan
class LoopUnrollBase : public OptimizerCaller {
 public:
  explicit LoopUnrollBase(bool need_check_primJ, bool need_process)
      : need_check_primJ_(need_check_primJ), need_process_(need_process) {}
  ~LoopUnrollBase() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> loop_func;
    PatternNode<AnfNodePtr> init;
    PatternNode<AnfNodePtr> xs;
    PatternNode<AnfNodePtr> length;
    PatternNode<AnfNodePtr> unroll;
    auto LoopUnrollLambda = [&node, &loop_func, &init, &xs, &length, &unroll]() -> AnfNodePtr {
      auto loop_func_node = loop_func.GetNode(node);
      auto init_node = init.GetNode(node);
      auto xs_node = xs.GetNode(node);
      auto length_node = length.GetNode(node);
      auto unroll_node = unroll.GetNode(node);
      auto fg = node->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      auto f_node = GetValueNode<FuncGraphPtr>(loop_func_node);
      MS_EXCEPTION_IF_NULL(f_node);
      // Unroll Directly
      bool is_none_node = IsValueNode<None>(xs_node);
      int64_t length_value = 0;
      if (IsValueNode<KeywordArg>(length_node)) {
        auto length_keyword_value = GetValueNode<KeywordArgPtr>(length_node)->get_value();
        MS_EXCEPTION_IF_NULL(length_keyword_value);
        auto length_int_ptr = length_keyword_value->cast<Int64ImmPtr>();
        MS_EXCEPTION_IF_NULL(length_int_ptr);
        length_value = length_int_ptr->value();
      } else {
        length_value = GetValueNode<Int64ImmPtr>(length_node)->value();
      }
      std::map<TypeId, PrimitivePtr> getitem_op_map = {{kObjectTypeTuple, prim::kPrimTupleGetItem},
                                                       {kObjectTypeList, prim::kPrimListGetItem},
                                                       {kObjectTypeDictionary, prim::kPrimDictGetItem},
                                                       {kMetaTypeNone, prim::kPrimTupleGetItem}};
      auto xs_abs = xs_node->abstract();
      MS_EXCEPTION_IF_NULL(xs_abs);
      MS_EXCEPTION_IF_NULL(xs_abs->GetType());
      auto type_id = xs_abs->GetType()->type_id();
      auto iter = getitem_op_map.find(type_id);
      PrimitivePtr getitem_op = iter->second;

      // Generate Loop FuncGraph for a single unrolled funcgraph call
      auto loop_unroll_fg = std::make_shared<FuncGraph>();
      AnfNodePtr loop_init = init_node;
      AnfNodePtr func_output = nullptr;
      AnfNodePtrList ys_result{NewValueNode(prim::kPrimMakeList)};
      for (int64_t i = 0; i < length_value; ++i) {
        AnfNodePtrList loop_inputs{{NewValueNode(f_node), loop_init}};
        if (is_none_node) {
          (void)loop_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
        } else {
          auto item =
            loop_unroll_fg->NewCNodeInOrder({NewValueNode(getitem_op), xs_node, NewValueNode(static_cast<int64_t>(i))});
          (void)loop_inputs.emplace_back(item);
        }
        func_output = loop_unroll_fg->NewCNodeInOrder(loop_inputs);
        loop_init = loop_unroll_fg->NewCNodeInOrder(
          {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(0))});
        auto new_y = loop_unroll_fg->NewCNodeInOrder(
          {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(1))});
        (void)ys_result.emplace_back(new_y);
      }
      auto loop_ys = loop_unroll_fg->NewCNodeInOrder(ys_result);
      auto output = loop_unroll_fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), loop_init, loop_ys});
      loop_unroll_fg->set_output(output);
      return fg->NewCNodeInOrder({NewValueNode(loop_unroll_fg)});
    };

    auto CheckNeedProcessScan = [this](const AnfNodePtr &node) -> bool {
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      // For pass after automatic differentiation
      if (need_process_) {
        return true;
      }
      // For pass before automatic differentiation, check kPrimJ for the first time
      if (need_check_primJ_) {
        auto fg = node->func_graph();
        MS_EXCEPTION_IF_NULL(fg);
        auto manager = fg->manager();
        MS_EXCEPTION_IF_NULL(manager);
        const auto &nodes = manager->all_nodes();
        for (auto &cur_node : nodes) {
          if (IsPrimitiveCNode(cur_node, prim::kPrimJ)) {
            MS_LOG(DEBUG) << "Need Process loop unroll before automatic differentiation pass.";
            set_need_check_primJ();
            return true;
          }
        }
        MS_LOG(DEBUG) << "Need Process loop unroll after automatic differentiation pass.";
        set_need_check_primJ();
        return false;
      }

      return false;
    };
    MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimScan, loop_func, init, xs, length, unroll), LoopUnrollLambda,
                            CheckNeedProcessScan(node));

    return nullptr;
  }

 private:
  bool need_check_primJ_;
  bool need_process_;
  void set_need_check_primJ() { need_check_primJ_ = false; }
};

class LoopUnrollBeforeGrad : public LoopUnrollBase {
 public:
  explicit LoopUnrollBeforeGrad(bool need_check_primJ = true, bool need_process = false)
      : LoopUnrollBase(need_check_primJ, need_process) {}
  ~LoopUnrollBeforeGrad() override = default;
};

class LoopUnrollAfterGrad : public LoopUnrollBase {
 public:
  explicit LoopUnrollAfterGrad(bool need_check_primJ = false, bool need_process = true)
      : LoopUnrollBase(need_check_primJ, need_process) {}
  ~LoopUnrollAfterGrad() override = default;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_LOOP_UNROLL_H_
