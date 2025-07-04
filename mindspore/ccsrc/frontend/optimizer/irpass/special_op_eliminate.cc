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

#include "frontend/optimizer/irpass/special_op_eliminate.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/utils.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "frontend/optimizer/pattern_matcher.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/comm_manager.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "utils/tensor_construct_utils.h"
#include "utils/ms_utils_secure.h"

namespace mindspore {
namespace opt {
namespace irpass {

SpecialOpEliminater::SpecialOpEliminater()
    : insert_gradient_of_(std::make_shared<PrimEliminater>(prim::kPrimInsertGradientOf)),
      stop_gradient_(std::make_shared<PrimEliminater>(prim::kPrimStopGradient)),
      hook_backward_(std::make_shared<PrimEliminater>(prim::kPrimHookBackward)),
      cell_backward_hook_(std::make_shared<PrimEliminater>(prim::kPrimCellBackwardHook)),
      print_shape_type_(std::make_shared<PrimEliminater>(prim::kPrimPrintShapeType)),
      mirror_(std::make_shared<PrimEliminater>(prim::kPrimMirror)),
      virtual_div_(std::make_shared<PrimEliminater>(prim::kPrimVirtualDiv)),
      mutable_(std::make_shared<PrimEliminater>(prim::kPrimMutable)) {
  (void)eliminaters_.emplace_back(insert_gradient_of_);
  (void)eliminaters_.emplace_back(stop_gradient_);
  (void)eliminaters_.emplace_back(hook_backward_);
  (void)eliminaters_.emplace_back(cell_backward_hook_);
  (void)eliminaters_.emplace_back(print_shape_type_);
  (void)eliminaters_.emplace_back(mirror_);
  (void)eliminaters_.emplace_back(virtual_div_);
  (void)eliminaters_.emplace_back(mutable_);
}

AnfNodePtr SpecialOpEliminater::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  AnfNodePtr new_node;
  for (auto &eliminater : eliminaters_) {
    new_node = (*eliminater)(optimizer, node);
    if (new_node != nullptr) {
      if (IsPrimitiveCNode(node, prim::kPrimHookBackward) || IsPrimitiveCNode(node, prim::kPrimCellBackwardHook)) {
        MS_LOG(WARNING) << "Hook operation does not work in graph mode or functions decorated with 'jit', it will be "
                           "eliminated during compilation.";
      }
      return new_node;
    }
  }
  return nullptr;
}

// {PrimVirtualDataset, X} -> X
// {PrimVirtualDataset, Xs} -> {prim::kPrimMakeTuple, Xs}
AnfNodePtr VirtualDatasetEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimVirtualDataset) || node->func_graph() == nullptr ||
      parallel::HasNestedMetaFg(node->func_graph())) {
    return nullptr;
  }

  auto &inputs = node->cast<CNodePtr>()->inputs();
  if (inputs.size() < 1) {
    return nullptr;
  }

  std::vector<AnfNodePtr> args;
  (void)std::copy(inputs.cbegin() + 1, inputs.cend(), std::back_inserter(args));
  (void)args.insert(args.cbegin(), NewValueNode(prim::kPrimMakeTuple));

  return node->func_graph()->NewCNode(args);
}

// {prim::kPrimVirtualOutput, X} -> X
AnfNodePtr VirtualOutputEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimVirtualOutput) || node->func_graph() == nullptr ||
      parallel::HasNestedMetaFg(node->func_graph())) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  return cnode->input(1);
}

// {prim::kPrimAShardIdentity, X} -> X
AnfNodePtr AShardIdentityEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimAShardIdentity) || node->func_graph() == nullptr ||
      parallel::HasNestedMetaFg(node->func_graph())) {
    return nullptr;
  }
  auto _cnode = node->cast<CNodePtr>();
  if (_cnode == nullptr) {
    return nullptr;
  }
  return _cnode->input(1);
}

// {ParallelVirtualNode, X, Y...} -> X
AnfNodePtr ParallelVirtualNodeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }
  auto input = cnode->input(1);
  if (input->isa<CNode>()) {
    auto input_cnode = input->cast<CNodePtr>();
    input_cnode->set_primal_attrs(cnode->primal_attrs());
  }
  return cnode->input(1);
}

// {prim::kPrimSameTypeShape, X, Y} -> X
AnfNodePtr SameEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  x_ = nullptr;
  AnfVisitor::Match(prim::kPrimSameTypeShape, {IsNode, IsNode})(node);
  return x_;
}

void SameEliminater::Visit(const AnfNodePtr &node) {
  if (x_ == nullptr) {
    x_ = node;
  }
}

// {prim::kPrimCheckBprop, X, Y} -> X
AnfNodePtr CheckBpropEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  x_ = nullptr;
  AnfVisitor::Match(prim::kPrimCheckBprop, {IsNode, IsNode})(node);
  return x_;
}

void CheckBpropEliminater::Visit(const AnfNodePtr &node) {
  if (x_ == nullptr) {
    x_ = node;
  }
}

// {prim::DumpGradient, X, Y, Z} -> Y
AnfNodePtr DumpGradientEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimDumpGradient) || node->func_graph() == nullptr) {
    return nullptr;
  }
  const CNodePtr dump_gradient_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dump_gradient_cnode);
  return dump_gradient_cnode->input(kIndex2);
}

// {prim::kPrimMiniStepAllGather, X, Z} -> {prim::kPrimAllGather, X}
AnfNodePtr MiniStepAllGatherPass::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMiniStepAllGather) || node->func_graph() == nullptr) {
    return nullptr;
  }

  auto &inputs = node->cast<CNodePtr>()->inputs();
  if (inputs.size() < 2) {
    return nullptr;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  std::string group = attrs[parallel::GROUP]->ToString();
  auto fusion = attrs[parallel::FUSION];
  bool contain_recompute = prim->HasAttr(parallel::RECOMPUTE);
  bool contain_segment = prim->HasAttr(parallel::SEGMENT);
  bool recompute = contain_recompute && GetValue<bool>(attrs[parallel::RECOMPUTE]);
  ValuePtr segment = contain_segment ? attrs[parallel::SEGMENT] : nullptr;
  parallel::Operator op = parallel::CreateAllGatherOp(group);
  std::vector<AnfNodePtr> node_input =
    parallel::CreateInput(op, inputs[1], parallel::PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE);
  auto prim_anf_node = node_input[0]->cast<ValueNodePtr>();
  prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  attrs = prim->attrs();
  attrs[parallel::FUSION] = fusion;
  if (contain_recompute) {
    attrs[parallel::RECOMPUTE] = MakeValue(recompute);
  }
  if (contain_segment) {
    attrs[parallel::SEGMENT] = segment;
  }
  (void)prim->SetAttrs(attrs);
  auto func_graph = inputs[1]->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_node = func_graph->NewCNode(node_input);
  if (node->cast<CNodePtr>()->HasPrimalAttr(kAttrSegment)) {
    new_node->AddPrimalAttr(kAttrSegment, node->cast<CNodePtr>()->GetPrimalAttr(kAttrSegment));
  }
  return new_node;
}

// {prim::kPrimMicroStepAllGather, X, Z} -> {prim::kPrimAllGather, X}
AnfNodePtr MicroStepAllGatherPass ::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMicroStepAllGather) || node->func_graph() == nullptr) {
    return nullptr;
  }

  auto &inputs = node->cast<CNodePtr>()->inputs();
  if (inputs.size() < 2) {
    return nullptr;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  std::string group = attrs[parallel::GROUP]->ToString();
  if (group.empty()) {
    return inputs[1];
  }
  auto fusion = attrs[parallel::FUSION];
  bool contain_recompute = prim->HasAttr(parallel::RECOMPUTE);
  bool contain_segment = prim->HasAttr(parallel::SEGMENT);
  bool recompute = contain_recompute && GetValue<bool>(attrs[parallel::RECOMPUTE]);
  ValuePtr segment = contain_segment ? attrs[parallel::SEGMENT] : nullptr;
  parallel::Operator op = parallel::CreateAllGatherOp(group);
  auto op_instance_name =
    recompute ? parallel::PARALLEL_OPTIMIZER_ALLGATHER : parallel::PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE;
  std::vector<AnfNodePtr> node_input = parallel::CreateInput(op, inputs[1], op_instance_name);
  auto prim_anf_node = node_input[0]->cast<ValueNodePtr>();
  prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  attrs = prim->attrs();
  attrs[parallel::FUSION] = fusion;
  if (contain_recompute) {
    attrs[parallel::RECOMPUTE] = MakeValue(recompute);
  }
  if (contain_segment) {
    attrs[parallel::SEGMENT] = segment;
  }
  (void)prim->SetAttrs(attrs);
  auto func_graph = inputs[1]->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_node = func_graph->NewCNode(node_input);
  return new_node;
}

// Reset defer_inline flag
AnfNodePtr ResetDeferInline::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (IsValueNode<FuncGraph>(node)) {
    auto fg = GetValueNode<FuncGraphPtr>(node);
    fg->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (RecomputeBeforeInline()) {
      fg->erase_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH);
      fg->erase_flag(FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH);
      fg->erase_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE);
    }
  }
  return nullptr;
}

// {PrimZerosLike, Y} -> {PrimFill, {PrimDType, Y}, {PrimShape, Y}, 0}
ZeroLikeFillZero::ZeroLikeFillZero() {
  py::gil_scoped_acquire gil;
  PrimFill_ = prim::GetPythonOps("fill", "mindspore.ops.functional")->cast<PrimitivePtr>();
  PrimShape_ = prim::GetPythonOps("shape_", "mindspore.ops.functional")->cast<PrimitivePtr>();
  PrimDType_ = prim::GetPythonOps("dtype", "mindspore.ops.functional")->cast<PrimitivePtr>();
}

AnfNodePtr ZeroLikeFillZero::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  y_ = nullptr;
  AnfVisitor::Match(prim::kPrimZerosLike, {IsNode})(node);
  if (y_ == nullptr || node->func_graph() == nullptr) {
    return nullptr;
  }
  if ((y_->abstract() == nullptr) || !y_->abstract()->isa<abstract::AbstractTensor>()) {
    auto fg = node->func_graph();
    auto dtype = fg->NewCNode({NewValueNode(PrimDType_), y_});
    auto shape = fg->NewCNode({NewValueNode(PrimShape_), y_});
    return fg->NewCNode({NewValueNode(PrimFill_), dtype, shape, NewValueNode(MakeValue(static_cast<int64_t>(0)))});
  }
  const auto &node_abs = node->abstract();
  if (node_abs != nullptr &&
      (node_abs->isa<abstract::AbstractRefTensor>() || node_abs->inplace_abstract() != nullptr)) {
    return nullptr;
  }

  abstract::AbstractTensorPtr tensor_abstract = y_->abstract()->cast<abstract::AbstractTensorPtr>();
  TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
  std::vector<int64_t> tensor_shape = tensor_abstract->shape()->shape();

  // if shape is unknown, don't optimize this operator away
  auto is_shape_unknown =
    std::any_of(tensor_shape.begin(), tensor_shape.end(), [](const auto &dimension) { return (dimension < 0); });
  if (is_shape_unknown) {
    return node;
  }

  tensor::TensorPtr new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
  size_t mem_size = GetTypeByte(tensor_type_ptr) * LongToSize(new_tensor_ptr->ElementsNum());
  uint8_t *data = reinterpret_cast<uint8_t *>(new_tensor_ptr->data_c());
  if (common::huge_memset(data, mem_size, 0x0, mem_size) != EOK) {
    MS_LOG(ERROR) << "For 'ZeroLikeFillZero', failed to init data memory.";
    return nullptr;
  }

  auto new_cnode = NewValueNode(new_tensor_ptr);
  new_cnode->set_abstract(new_tensor_ptr->ToAbstract());

  return new_cnode;
}

void ZeroLikeFillZero::Visit(const AnfNodePtr &node) { y_ = node; }

// {prim::kPrimDepend, X, ValueCond} -> X
// {prim::kPrimDepend, {prim, X, ...}, X} -> {prim, X, ...}
AnfNodePtr DependValueElim::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> x;
  PatternNode cond;
  PatternNode x_user;
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimDepend, x, cond), x, IsVNode(cond.GetNode(node)));
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimDepend, x_user, x), x_user,
                   IsUsedByOther(x.GetNode(node), x_user.GetNode(node)));
  return nullptr;
}

bool DependValueElim::IsUsedByOther(const AnfNodePtr &node, const AnfNodePtr &user_node) const {
  if (!user_node->isa<CNode>()) {
    return false;
  }
  auto user_cnode = user_node->cast<CNodePtr>();
  auto inputs = user_cnode->inputs();
  return std::any_of(inputs.begin(), inputs.end(), [&node](const AnfNodePtr &input) { return input == node; });
}

// {{prim:getattr, {prim::resolve, SymbolStr, C}, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {prim:getattr, {prim::resolve, SymbolStr, zeros_like}, Xy} ->Tensor(0, shape(Xy))
// {{prim::resolve, CommonOPS, getitem}, (tensor0, tensor1,...), 0} -> tensor0
bool PynativeEliminater::CheckNameSpaceVNode(const AnfNodePtr &node, const std::string &str_value) const {
  ValueNodePtr value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return false;
  }
  auto name_space = GetValueNode<parse::NameSpacePtr>(value_node);
  MS_EXCEPTION_IF_NULL(name_space);
  auto module_name = name_space->module();
  return module_name.find(str_value) != std::string::npos;
}

bool PynativeEliminater::CheckSymbolVNode(const AnfNodePtr &node, const std::string &str_value) const {
  ValueNodePtr value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return false;
  }
  auto symbol = GetValueNode<parse::SymbolPtr>(value_node);
  MS_EXCEPTION_IF_NULL(symbol);
  return symbol->symbol() == str_value;
}
bool PynativeEliminater::CheckStrVNode(const AnfNodePtr &node, const std::string &str_value) const {
  ValueNodePtr value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return false;
  }
  auto string_imm = GetValueNode<StringImmPtr>(value_node);
  MS_EXCEPTION_IF_NULL(string_imm);
  return string_imm->value() == str_value;
}

ValuePtr PynativeEliminater::FillGetItem(const ValuePtr &value, const ValuePtr &idx, const AnfNodePtr &node) const {
  MS_LOG(DEBUG) << "Start FillGetItem" << value->ToString() << idx->ToString();
  if (!idx->isa<Int64Imm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Getitem idx must int:" << idx->ToString();
  }

  if (!value->isa<ValueTuple>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Getitem value must tuple:" << value->ToString();
  }

  auto value_tuple = value->cast<ValueTuplePtr>();
  int idx_t = idx->cast<Int64ImmPtr>()->value();
  MS_LOG(DEBUG) << "Fill getitem" << idx_t << (*value_tuple)[idx_t]->ToString();
  return (*value_tuple)[idx_t];
}

ValuePtr PynativeEliminater::FillZero(const ValuePtr &value, const AnfNodePtr &node) {
  MS_LOG(DEBUG) << "Start FillZero";
  ValuePtr out = nullptr;
  if (value->isa<Int64Imm>()) {
    return MakeValue(value->cast<Int64ImmPtr>()->value());
  }

  if (value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "Start FillZero Tensor";
    auto tensor = value->cast<tensor::TensorPtr>();
    auto out_t = TensorConstructUtils::CreateZerosTensor(tensor->Dtype(), tensor->shape());
    MS_EXCEPTION_IF_NULL(out_t);
    char *data = reinterpret_cast<char *>(out_t->data_c());
    std::fill(data, data + out_t->data().nbytes(), 0);
    out = out_t;
  }

  std::vector<ValuePtr> value_list;
  if (value->isa<ValueTuple>()) {
    MS_LOG(DEBUG) << "Start FillZero Tuple" << value->ToString();
    auto value_tuple = value->cast<ValueTuplePtr>();
    for (size_t i = 0; i < value_tuple->size(); i++) {
      value_list.push_back(FillZero((*value_tuple)[i], node));
    }
    out = std::make_shared<ValueTuple>(value_list);
  }
  if (out == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "FillZero failed:" << value->ToString();
  }
  MS_LOG(DEBUG) << "Result: " << out->ToString();
  return out;
}

AnfNodePtr PynativeEliminater::OperatorHandle1(const PatternNode<AnfNodePtr> &arg, const AnfNodePtr &node) {
  auto rep = (arg).GetNode(node);
  if (rep != nullptr) {
    if (rep->isa<ValueNode>()) {
      auto value_node = rep->cast<ValueNodePtr>();
      auto new_value_node = NewValueNode(FillZero(value_node->value(), node));
      new_value_node->set_has_new_value(value_node->has_new_value());
      MS_LOG(DEBUG) << "Zeros_like replace ok " << rep->DebugString(4);
      return new_value_node;
    }
  }
  return nullptr;
}

AnfNodePtr PynativeEliminater::OperatorHandle2(const PatternNode<AnfNodePtr> &arg, const AnfNodePtr &node) {
  auto rep = (arg).GetNode(node);
  if (rep != nullptr) {
    if (rep->isa<ValueNode>() && !HasAbstractMonad(rep)) {
      auto value_node = rep->cast<ValueNodePtr>();
      auto new_value_node = NewValueNode(FillZero(value_node->value(), node));
      new_value_node->set_has_new_value(value_node->has_new_value());
      MS_LOG(DEBUG) << "Zeros_like replace ok 2 " << rep->DebugString(4);
      return new_value_node;
    }
  }
  return nullptr;
}

void PynativeEliminater::OperatorHandle3(const std::vector<PatternNode<AnfNodePtr>> &args,
                                         const AnfNodePtr &node) const {
  constexpr size_t args_size = 2;
  for (size_t i = 0; i < args_size; i++) {
    auto rep = (args[i]).GetNode(node);
    if (rep != nullptr && rep->isa<ValueNode>()) {
      auto value_node = rep->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto &value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      // when the use count of value node equals to one, it only used in binop_grad_common function
      if (value->isa<tensor::Tensor>() && value_node->used_graph_count() == 1) {
        auto tensor = value->cast<tensor::TensorPtr>();
        MS_EXCEPTION_IF_NULL(tensor);
        auto new_tensor = std::make_shared<tensor::Tensor>(tensor->Dtype()->type_id(), tensor->shape());
        value_node->set_value(new_tensor);
      }
    }
  }
}

AnfNodePtr PynativeEliminater::OperatorHandle4(const PatternNode<AnfNodePtr> &arg, const PatternNode<AnfNodePtr> &arg1,
                                               const AnfNodePtr &node) const {
  auto rep = (arg).GetNode(node);
  if (rep != nullptr) {
    if (rep->isa<ValueNode>()) {
      MS_LOG(DEBUG) << "Rep is " << rep->DebugString(4);
      ValueNodePtr new_node;
      auto value_node = rep->cast<ValueNodePtr>();
      auto rep1 = (arg1).GetNode(node);
      if (rep1 != nullptr) {
        if (rep1->isa<ValueNode>()) {
          auto idx = rep1->cast<ValueNodePtr>();
          if (!value_node->value()->isa<ValueTuple>()) {
            return nullptr;
          }
          new_node = NewValueNode(FillGetItem(value_node->value(), idx->value(), node));
          new_node->set_has_new_value(value_node->has_new_value());
        }
      }
      MS_LOG(DEBUG) << "Fill getitem  replace ok " << new_node->DebugString(4);
      return new_node;
    }
  }
  return nullptr;
}

AnfNodePtr PynativeEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  MS_LOG(DEBUG) << "Start replace node " << node->DebugString(4);
  PatternNode<AnfNodePtr> symbol_str_vnode;
  PatternNode<AnfNodePtr> c_vnode;
  PatternNode<AnfNodePtr> zeros_like_vnode;
  PatternNode<AnfNodePtr> arg;
  auto resolve = PPrimitive(prim::kPrimResolve, symbol_str_vnode, c_vnode);
  auto getattr = PPrimitive(prim::kPrimGetAttr, resolve, zeros_like_vnode);
  auto pattern = PCNode(getattr, arg);
  // {{prim:getattr, {prim::resolve, SymbolStr, C}, zeros_like}, Xy} ->Tensor(0, shape(Xy))
  if ((pattern).TryCapture(node) &&
      (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "SymbolStr") &&
       CheckSymbolVNode(c_vnode.GetNode(node), "C") && CheckStrVNode(zeros_like_vnode.GetNode(node), "zeros_like"))) {
    auto new_value_node = OperatorHandle1(arg, node);
    if (new_value_node != nullptr) {
      return new_value_node;
    }
  }
  MS_LOG(DEBUG) << "End replace 1 " << node->DebugString(4);
  // {prim:getattr, {prim::resolve, SymbolStr, zeros_like}, Xy} ->Tensor(0, shape(Xy))
  auto resolve1 = PPrimitive(prim::kPrimResolve, symbol_str_vnode, zeros_like_vnode);
  auto pattern1 = PCNode(resolve1, arg);

  if ((pattern1).TryCapture(node) && (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "SymbolStr") &&
                                      CheckSymbolVNode(zeros_like_vnode.GetNode(node), "zeros_like"))) {
    auto new_value_node = OperatorHandle2(arg, node);
    if (new_value_node != nullptr) {
      return new_value_node;
    }
  }
  // {prim:getattr, {prim::resolve, SymbolStr, binop_grad_common}, x, y, out, dout} -> {shape(x), shape(y), out, dout}
  PatternNode<AnfNodePtr> binop_grad_common;
  PatternNode<AnfNodePtr> getitem_vnode;
  std::vector<PatternNode<AnfNodePtr>> args(4);
  auto resolve_binop = PPrimitive(prim::kPrimResolve, symbol_str_vnode, binop_grad_common);
  auto pattern_binop = PCNode(resolve_binop, args[0], args[1], args[2], args[3]);
  if ((pattern_binop).TryCapture(node) && (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "SymbolStr") &&
                                           CheckSymbolVNode(binop_grad_common.GetNode(node), "binop_grad_common"))) {
    OperatorHandle3(args, node);
    return nullptr;
  }
  // resolve(CommonOPS, getitem)((tensors), 3)
  PatternNode<AnfNodePtr> arg1;
  auto resolve2 = PPrimitive(prim::kPrimResolve, symbol_str_vnode, getitem_vnode);
  auto pattern2 = PCNode(resolve2, arg, arg1);
  if ((pattern2).TryCapture(node) && (CheckNameSpaceVNode(symbol_str_vnode.GetNode(node), "CommonOPS") &&
                                      CheckSymbolVNode(getitem_vnode.GetNode(node), "getitem"))) {
    auto new_value_node = OperatorHandle4(arg, arg1, node);
    if (new_value_node != nullptr) {
      return new_value_node;
    }
  }

  MS_LOG(DEBUG) << "End Replace " << node->DebugString(4);
  return nullptr;
}

AnfNodePtr AllReduceConstElim::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> x;
  auto pattern = PPrimitive(prim::kPrimAllReduce, x);
  // If AllReduce takes constant value as input and values across devices are all the same(ensured by parallel mode)
  if (pattern.TryCapture(node) && IsVNode(x.GetNode(node)) &&
      parallel::IsAutoParallelCareGraph(pattern.GetFuncGraph())) {
    auto cur_func_graph = pattern.GetFuncGraph();
    // If reduce operation is sum, then multiply constant by number of devices, otherwise just return the constant
    auto prim_cnode = pattern.GetOriginalNode();
    MS_EXCEPTION_IF_NULL(prim_cnode);
    auto primitive = GetCNodePrimitive(prim_cnode);
    MS_EXCEPTION_IF_NULL(primitive);
    auto reduce_op = primitive->GetAttr("op");
    MS_EXCEPTION_IF_NULL(reduce_op);
    auto reduce_group = primitive->GetAttr("group");
    MS_EXCEPTION_IF_NULL(reduce_group);
    auto group = reduce_group->ToString();
    // For sum operation, multiply constant tensor by number of devices
    if (reduce_op->ToString() == "sum") {
      uint32_t num_of_devices;
      // Get number of devices
      if (!CommManager::GetInstance().GetRankSize(group, &num_of_devices)) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "Failed to get num of devices for group [" + group + "]";
      }
      // Multiply constant by number of devices then return
      std::vector<AnfNodePtr> mul_inputs;
      auto constant_node = x.GetNode(node);
      MS_EXCEPTION_IF_NULL(constant_node);
      auto constant_value_node = constant_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(constant_value_node);
      if (!constant_value_node->value()->isa<tensor::Tensor>()) {
        MS_LOG_WITH_NODE(EXCEPTION, constant_value_node)
          << "Expect the constant input for AllReduce to be a Tensor. Got " + constant_value_node->value()->ToString();
      }
      auto constant_tensor = constant_value_node->value()->cast<tensor::TensorPtr>();
      auto tensor_dtype = constant_tensor->Dtype();
      auto num_of_device_node =
        NewValueNode(std::make_shared<tensor::Tensor>(static_cast<int64_t>(num_of_devices), tensor_dtype));
      // Multiply nodes
      auto mul_prim = prim::GetPythonOps("tensor_mul", "mindspore.ops.functional");
      MS_EXCEPTION_IF_NULL(mul_prim);
      mul_inputs.push_back(NewValueNode(mul_prim));
      mul_inputs.push_back(constant_node);
      mul_inputs.push_back(num_of_device_node);
      return cur_func_graph->NewCNode(mul_inputs);
    } else {
      return x.GetNode(node);
    }
  }
  return nullptr;
}

// This pattern introduced by Depend(CollectCNodeWithIsolateNodes) in program_specialize.cc
// {{prim::kPrimDepend, X, Y}, Xs}->{prim::kPrimDepend, {X, Xs}, Y}
AnfNodePtr FloatDependGCall::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!node->isa<CNode>() || node->func_graph() == nullptr) {
    return nullptr;
  }
  auto &inputs = node->cast<CNodePtr>()->inputs();
  // as IsCNodeDup had checked the size of inputs must be greater or equal than 1, so no check here.
  if (IsPrimitiveCNode(inputs[0], prim::kPrimDepend)) {
    auto &depend_inputs = inputs[0]->cast<CNodePtr>()->inputs();
    constexpr auto number_three = 3;
    constexpr auto number_two = 2;
    if (depend_inputs.size() != number_three) {
      return nullptr;
    }
    // put {Y, Xs} to new_inputs;
    std::vector<AnfNodePtr> new_inputs({depend_inputs[1]});
    (void)new_inputs.insert(new_inputs.cend(), inputs.cbegin() + 1, inputs.cend());
    TraceGuard guard(MakeTraceInfo<TraceCopy>(node->debug_info()));
    ScopePtr scope = node->scope();
    ScopeGuard scope_guard(scope);
    auto new_call_node = node->func_graph()->NewCNode(new_inputs);
    auto new_node = node->func_graph()->NewCNode({depend_inputs[0], new_call_node, depend_inputs[number_two]});
    const auto &abs = node->abstract();
    if (abs != nullptr) {
      new_node->set_abstract(abs);
    }
    return new_node;
  }
  return nullptr;
}

AnfNodePtr PynativeGradjitPrimitivePyEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Start eliminate PrimitvePy for node: " << node->DebugString();
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &prim = GetCNodePrimitive(cnode);
  if (!prim || !prim->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "Node is not a primitivepy, mismatch: " << node->DebugString();
  }
  std::vector<AnfNodePtr> args = {NewValueNode(std::make_shared<Primitive>(*prim))};
  const auto &inputs = cnode->inputs();
  (void)std::copy(inputs.cbegin() + 1, inputs.cend(), std::back_inserter(args));
  const auto &func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  return func_graph->NewCNode(args);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
