/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "include/common/symbol_engine/symbol_engine_impl.h"
#include <algorithm>
#include <ostream>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/graph_utils.h"
#include "abstract/abstract_function.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/infer/symbol_ops_impl/switch.h"
#include "mindspore/ops/infer/symbol_ops_impl/j_op.h"
#include "utils/check_convert_utils.h"
#include "utils/anf_utils.h"
#include "symbolic_shape/utils.h"
#include "symbolic_shape/operation_builder.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/ccsrc/include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace symshape {
std::pair<FuncGraphPtr, size_t> GetFuncGraphFromCNode(const CNodePtr &cnode) {
  auto sub_fg = GetCNodeFuncGraph(cnode);
  size_t begin_index = kIndex1;
  if (sub_fg == nullptr && IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
    auto vnode = cnode->input(kIndex1)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(vnode);
    sub_fg = vnode->value()->cast<FuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(sub_fg);
    begin_index = kIndex2;
  }
  if (sub_fg == nullptr && common::AnfAlgo::HasNodeAttr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, cnode) &&
      common::AnfAlgo::HasNodeAttr(kAttrFuncGraph, cnode)) {
    sub_fg = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(cnode, kAttrFuncGraph);
  }
  if (sub_fg != nullptr && sub_fg->parameters().size() + begin_index < cnode->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For graph " << sub_fg->ToString() << ", the parameter size "
                               << sub_fg->parameters().size() << " is less than cnode input num "
                               << cnode->size() - begin_index;
  }
  return std::make_pair(sub_fg, begin_index);
}

class ControlFlowJoinNode : public SpecialCNodeHelper {
 public:
  using SpecialCNodeHelper::SpecialCNodeHelper;
  static bool Match(const CNodePtr &cnode) { return IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch); }
  void SetDependStatus(std::map<AnfNodePtr, DependStatus> *depend_status_map) override {
    auto input0 = input();
    (*depend_status_map)[input0->input(kIndex1)].value = true;
    SetFuncGraphDepend(input0->input(kIndex2));
    SetFuncGraphDepend(input0->input(kIndex3));
  }
  std::pair<PrimitivePtr, AbstractBasePtrList> ExtractInputs() override {
    auto prim = std::make_shared<Primitive>(ops::kControlFlowJoin);
    AbstractBasePtrList inputs;
    auto input0 = input();
    (void)inputs.emplace_back(input0->input(kIndex1)->abstract());
    (void)inputs.emplace_back(GetFuncGraphOutAbs(input0->input(kIndex2)));
    (void)inputs.emplace_back(GetFuncGraphOutAbs(input0->input(kIndex3)));
    return std::make_pair(std::move(prim), std::move(inputs));
  }

 protected:
  CNodePtr input() const {
    auto input0 = cnode_->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input0);
    return input0;
  }
  SymbolEngineImplPtr symbol_engine() const {
    auto symbol_engine = cnode_->func_graph()->symbol_engine();
    MS_EXCEPTION_IF_NULL(symbol_engine);
    auto symbol_engine_impl = symbol_engine->cast<SymbolEngineImplPtr>();
    MS_EXCEPTION_IF_NULL(symbol_engine_impl);
    return symbol_engine_impl;
  }
  void SetFuncGraphDepend(const AnfNodePtr &node) const {
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg != nullptr) {
      symbol_engine()->PreBuildQuerySubgraphDependStatus(cnode_, fg, kIndex1);
    }
  }

  AbstractBasePtr GetFuncGraphOutAbs(const AnfNodePtr &node) const {
    if (IsPrimitiveCNode(node, prim::kPrimPartial)) {
      return GetFuncGraphFromCNode(node->cast<CNodePtr>()).first->output()->abstract();
    }
    // the graph with Partial is build symbols ahead, build the pure graph (without Partial) in Switch.
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg == nullptr) {
      MS_EXCEPTION_IF_NULL(node->abstract());
      return node->abstract();
    }
    symbol_engine()->BuildSubgraphImpl(cnode_, fg, kIndex1);
    return fg->output()->abstract();
  }
};

class JFuncCaller : public SpecialCNodeHelper {
 public:
  /// \brief The call node of PrimJ:
  ///
  ///  %0 = J(@fg) // primitive "J"
  ///  %1 = %0(inp1, inp2, ...) // the node output a tuple of "(tensor, Func)"
  ///  %2 = TupleGetItem(%1, 1)  // get the output "Func"
  ///  %3 = %2(loss_scale)       // call the "Func".
  ///
  /// this pattern match the "%3", and the output shape is same as "inp1, inp2, ...".
  explicit JFuncCaller(const CNodePtr &cnode) : SpecialCNodeHelper(cnode) {
    auto getitem1 = cnode->input(kIndex0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(getitem1);
    input_ = getitem1->input(kIndex1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_);
  }
  ~JFuncCaller() override = default;
  static bool Match(const CNodePtr &cnode) {
    auto getitem1 = cnode->input(kIndex0)->cast<CNodePtr>();
    if (getitem1 == nullptr || !IsPrimitiveCNode(getitem1, prim::kPrimTupleGetItem)) {
      return false;
    }
    if (GetValue<int64_t>(GetValueNode(getitem1->input(kIndex2))) != 1) {
      return false;
    }
    auto callj = getitem1->input(kIndex1)->cast<CNodePtr>();
    return callj != nullptr && IsPrimitiveCNode(callj->input(kIndex0), prim::kPrimJ);
  }
  void SetDependStatus(std::map<AnfNodePtr, DependStatus> *depend_status_map) override {
    for (size_t i = 1; i < input_->size(); i++) {
      (*depend_status_map)[input_->input(i)] = (*depend_status_map)[cnode_];
    }
  }
  std::pair<PrimitivePtr, AbstractBasePtrList> ExtractInputs() override {
    auto prim = std::make_shared<Primitive>(ops::kJFuncCaller);
    AbstractBasePtrList inputs;
    inputs.reserve(input_->size());
    (void)std::transform(input_->inputs().begin(), input_->inputs().end(), std::back_inserter(inputs),
                         [](const AnfNodePtr &node) { return node->abstract(); });
    return std::make_pair(std::move(prim), std::move(inputs));
  }

 protected:
  CNodePtr input_{nullptr};
};

SymbolEngineImplPtr SymbolEngineImpl::Build(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Start to build symbol engine for func_graph [" << func_graph->ToString() << "].";
  SymbolEngineImplPtr engine = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    if (func_graph->symbol_engine() != nullptr) {
      CleanSymbols(func_graph);
    }
    engine = std::make_shared<SymbolEngineImpl>(func_graph);
    func_graph->set_symbol_engine(engine);
    engine->PreBuild();
    engine->BuildImpl();
    MS_LOG(INFO) << "Build symbol engine for func_graph [" << func_graph->ToString() << "]successfully.";
  } catch (std::exception &e) {
    if (engine != nullptr) {
      engine->CleanBuildingTmp();
    }
    MS_LOG(WARNING) << "A problem occurs when building symbol engine for func_graph [" << func_graph->ToString()
                    << "]: " << e.what();
    return nullptr;
  }
  return engine;
}

void SymbolEngineImpl::BuildNodesSymbol(const FuncGraphPtr &fg, const AnfNodePtrList &cnodes) {
  for (auto &node : cnodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (auto fg_with_index = GetFuncGraphFromCNode(cnode); fg_with_index.first != nullptr) {
      // "call" or "Partial" node
      BuildSubgraphImpl(cnode, fg_with_index.first, fg_with_index.second);
      // For "call" node, after building graph, if the output symbol is none, it means the graph has not been fully
      // built. It may be a loop body. In this case, we should not build the following nodes from this graph.
      if (fg_with_index.second == kIndex1) {
        auto sub_fg_abs = fg_with_index.first->output()->abstract();
        if (sub_fg_abs->GetSymbolicShape() == nullptr && sub_fg_abs->GetSymbolicValue() == nullptr) {
          MS_LOG(DEBUG) << "Early stop building symbols for " << fg->ToString() << ", because the symbols of subgraph "
                        << fg_with_index.first->ToString() << " has not been fully built.";
          break;
        }
      }
    } else {
      BuildCNodeSymbol(cnode);
    }
  }
  // the funcgraph can be empty or only return a ValueNode.
  if (!cnodes.empty()) {
    return;
  }
  auto node = fg->output();
  if (node->isa<ValueNode>()) {
    auto depend_status = depend_status_map_[node];
    auto node_abs = CloneAbstractIfSymbolExists(node);
    MS_EXCEPTION_IF_NULL(node_abs);
    if (depend_status.shape) {
      auto sym_shape = node_abs->GetShape()->BuildSymbolicShape();
      MS_LOG(DEBUG) << "Set shape for node: " << node->DebugString() << ". symbol: " << sym_shape->ToString();
      node_abs->SetSymbolicShape(sym_shape);
    }
    if (depend_status.value) {
      auto sym_value = BuildSymbolicValue(node_abs);
      MS_LOG(DEBUG) << "Set value for node: " << node->DebugString() << ". symbol: " << sym_value->ToString();
      node_abs->SetSymbolicValue(sym_value);
    }
  }
}

void SymbolEngineImpl::PreBuild() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Prebuild " << ToString() << " with graph " << func_graph->ToString();
  visited_graph_.clear();
  visited_graph_[func_graph.get()] = 1;
  GetAllNodes(func_graph);
  PreBuildQueryDependStatus(GetCNodesOfFuncGraph(func_graph));
}

void SymbolEngineImpl::BuildImpl() {
  auto func_graph = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "Build " << ToString() << " with graph " << func_graph->ToString();
  emitter_ = std::make_unique<OperationEmitter>(&ops_);
  visited_graph_.clear();
  visited_graph_[func_graph.get()] = 1;
  BuildNodesSymbol(func_graph, GetCNodesOfFuncGraph(func_graph));
  CleanBuildingTmp();
}

void SymbolEngineImpl::CleanBuildingTmp() {
  emitter_->Clean();
  visited_graph_.clear();
  generalized_shape_.clear();
  generalized_value_.clear();
  fg_cnodes_.clear();
}

void SymbolEngineImpl::PreBuildSpecialNode(const CNodePtr &cnode) {
  std::shared_ptr<SpecialCNodeHelper> helper = nullptr;
  if (ControlFlowJoinNode::Match(cnode)) {
    helper = std::make_shared<ControlFlowJoinNode>(cnode);
  } else if (JFuncCaller::Match(cnode)) {
    helper = std::make_shared<JFuncCaller>(cnode);
  } else {
    MS_LOG(DEBUG) << "The special node " << cnode->fullname_with_scope() << " is not supported.";
    return;
  }
  special_cnodes_[cnode] = helper;
  helper->SetDependStatus(&depend_status_map_);
}

void SymbolEngineImpl::SetInputDependStatus(const CNodePtr &cnode, bool depend_value) {
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  size_t input_num = cnode->size() - 1;
  auto depends = depend_value ? GetValueDepends(prim, input_num) : GetShapeDepends(prim, input_num);
  for (size_t i = 0; i < depends.size(); i++) {
    if (depends[i] == DependOn::kValue) {
      depend_status_map_[cnode->input(i + 1)].value = true;
    } else if (depends[i] == DependOn::kShape) {
      depend_status_map_[cnode->input(i + 1)].shape = true;
    }
  }
}

void SymbolEngineImpl::PreBuildQueryDependStatus(const AnfNodePtrList &cnodes) {
  for (auto iter = cnodes.rbegin(); iter != cnodes.rend(); ++iter) {
    auto cnode = (*iter)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &depend_status = depend_status_map_[cnode];
    if (!depend_status.value && !depend_status.shape) {
      // build symbolic shape for the node even though it's not depended by any nodes.
      depend_status.shape = true;
    }
    MS_LOG(DEBUG) << "The depend status of " << cnode->DebugString() << "(" << cnode->fullname_with_scope()
                  << "): shape-depend=" << depend_status.shape << ", value-depend=" << depend_status.value;
    if (cnode->input(0)->isa<CNode>()) {
      PreBuildSpecialNode(cnode);
      continue;
    }
    // the "call" node or Partial node.
    auto subfg_with_index = GetFuncGraphFromCNode(cnode);
    if (subfg_with_index.first != nullptr) {
      PreBuildQuerySubgraphDependStatus(cnode, subfg_with_index.first, subfg_with_index.second);
      continue;
    }
    // the normal CNode, check the depend status from operation builder info.
    if (!OperationBuilderInfoRegistry::HasOp(AnfUtils::GetCNodeName(cnode))) {
      continue;
    }
    if (depend_status.shape) {
      SetInputDependStatus(cnode, false);
    }
    if (depend_status.value) {
      SetInputDependStatus(cnode, true);
    }
  }
}

void SymbolEngineImpl::PreBuildQuerySubgraphDependStatus(const CNodePtr &cnode, const FuncGraphPtr &sub_fg,
                                                         size_t begin_input_index) {
  if (++visited_graph_[sub_fg.get()] > 1) {
    return;
  }
  sub_fg->set_symbol_engine(shared_from_base<SymbolEngine>());
  depend_status_map_[sub_fg->output()] = depend_status_map_[cnode];
  PreBuildQueryDependStatus(GetCNodesOfFuncGraph(sub_fg));
  for (auto &param : sub_fg->parameters()) {
    if (begin_input_index >= cnode->size()) {
      break;
    }
    auto &cnode_input_depend_status = depend_status_map_[cnode->input(begin_input_index++)];
    auto depend_status = depend_status_map_[param];
    if (depend_status.shape) {
      cnode_input_depend_status.shape = true;
    }
    if (depend_status.value) {
      cnode_input_depend_status.value = true;
    }
  }
}

bool SymbolEngineImpl::Infer(const AbstractBasePtrList &inputs) {
  if (!support_infer_) {
    MS_LOG(WARNING) << "The " << ToString() << " does not support infer";
    return false;
  }
  MS_LOG(DEBUG) << "Infer " << ToString() << " with inputs: " << inputs;
  auto fg = func_graph_.lock();
  MS_EXCEPTION_IF_NULL(fg);
  auto &params = fg->parameters();
  // There may be params like UpdateStates, which won't contribute to infer
  if (params.size() < inputs.size()) {
    MS_LOG(EXCEPTION) << "The parameter size should be equal to or larger than inputs size, but got " << params.size()
                      << " vs " << inputs.size();
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (auto shape = params[i]->abstract()->GetSymbolicShape(); shape != nullptr) {
      auto cur_shape = inputs[i]->GetShape()->BuildSymbolicShape();
      MS_EXCEPTION_IF_NULL(cur_shape);
      MS_LOG(DEBUG) << "Update shape for input[" << i << "]: " << cur_shape->ToRawString();
      shape->Update(cur_shape);
    }
    if (auto value = params[i]->abstract()->GetSymbolicValue(); value != nullptr && value->CanUpdate()) {
      auto cur_value = BuildSymbolicValue(inputs[i]);
      MS_EXCEPTION_IF_NULL(cur_value);
      MS_LOG(DEBUG) << "Update value for input[" << i << "]: " << cur_value->ToRawString();
      value->Update(cur_value);
    }
  }
  for (auto &op : ops_) {
    op->Run();
  }
  return true;
}

bool SymbolEngineImpl::IsDependValue(const AnfNodePtr &node) {
  if (depend_status_map_.find(node) != depend_status_map_.end()) {
    return depend_status_map_[node].value;
  }
  return false;
}

bool SymbolEngineImpl::IsDependShape(const AnfNodePtr &node) {
  if (depend_status_map_.find(node) != depend_status_map_.end()) {
    return depend_status_map_[node].shape;
  }
  return false;
}

bool SymbolEngineImpl::GeneralizeParamShape(const AnfNodePtr &param, const AbstractBasePtr &input_abs) {
  if (generalized_shape_.count(param) > 0) {
    return false;
  }
  auto param_abs = param->abstract();
  MS_EXCEPTION_IF_NULL(param_abs);
  if (param_abs->GetSymbolicShape() == nullptr || input_abs->GetSymbolicShape() == nullptr) {
    return false;
  }
  auto param_shape = param_abs->GetSymbolicShape();
  auto input_shape = input_abs->GetSymbolicShape();
  if (param_shape->EqualsTo(input_shape)) {
    return false;
  }
  bool build_again = false;
  bool gen_all = false;
  std::function<SymbolPtrList(const SymbolPtrList &, const SymbolPtrList &)> process;
  process = [&build_again, &gen_all, &process](const SymbolPtrList &ori_sym, const SymbolPtrList &new_sym) {
    SymbolPtrList ret;
    if (ori_sym.size() != new_sym.size()) {
      gen_all = true;
      return ret;
    }
    ret = ori_sym;
    for (size_t i = 0; i < ori_sym.size(); i++) {
      if (ori_sym[i]->EqualsTo(new_sym[i])) {
        continue;
      }
      if (ori_sym[i]->is<ListSymbol>()) {
        ret[i] =
          ListSymbol::Make(process(ori_sym[i]->as<ListSymbol>()->symbols(), new_sym[i]->as<ListSymbol>()->symbols()));
        continue;
      }
      auto gen_symbol = ops::JoinIntSymbol(ori_sym[i], new_sym[i]);
      if (gen_symbol != ori_sym[i]) {
        build_again = true;
        ret[i] = gen_symbol;
      }
    }
    return ret;
  };
  auto ret = process(param_shape->symbols(), input_shape->symbols());
  if (gen_all) {
    (void)generalized_shape_.insert(param);
    param_abs->SetSymbolicShape(param_abs->GetShape()->BuildSymbolicShape());
    return true;
  }
  return build_again;
}

bool SymbolEngineImpl::GeneralizeParamValue(const AnfNodePtr &param, const AbstractBasePtr &input_abs) {
  if (generalized_value_.count(param) > 0) {
    return false;
  }
  auto param_abs = param->abstract();
  MS_EXCEPTION_IF_NULL(param_abs);
  if (param_abs->GetSymbolicValue() == nullptr || input_abs->GetSymbolicValue() == nullptr) {
    return false;
  }
  auto param_value = param_abs->GetSymbolicValue();
  auto input_value = input_abs->GetSymbolicValue();
  if (param_value->EqualsTo(input_value)) {
    return false;
  }
  param_abs->SetSymbolicValue(BuildSymbolicValue(param_abs));
  (void)generalized_value_.insert(param);
  return true;
}

bool SymbolEngineImpl::SetParamSymbols(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index,
                                       size_t visit_cnt) {
  bool build_again = false;
  const size_t max_visit_cnt = 5;  // to avoid unexplained dead loop
  for (size_t i = begin_input_index; i < cnode->size(); i++) {
    auto inp = cnode->input(i);
    auto input_abs = inp->abstract();
    MS_EXCEPTION_IF_NULL(input_abs);
    if (IsDependShape(inp) && input_abs->GetSymbolicShape() == nullptr) {
      input_abs->SetSymbolicShape(input_abs->GetShape()->BuildSymbolicShape());
    }
    if (IsDependValue(inp) && input_abs->GetSymbolicValue() == nullptr) {
      input_abs->SetSymbolicValue(BuildSymbolicValue(input_abs));
    }
    auto param = sub_fg->parameters()[i - begin_input_index];
    auto param_abs = param->abstract();
    if (visit_cnt == 1) {
      param_abs = CloneAbstractIfSymbolExists(param);
      MS_EXCEPTION_IF_NULL(param_abs);
      param_abs->SetSymbolicShape(input_abs->GetSymbolicShape());
      param_abs->SetSymbolicValue(input_abs->GetSymbolicValue());
    } else if (visit_cnt <= max_visit_cnt) {
      build_again = GeneralizeParamShape(param, input_abs) || build_again;
      build_again = GeneralizeParamValue(param, input_abs) || build_again;
    }
    MS_LOG(DEBUG) << "Symbol of param[" << i - begin_input_index << "]: S:"
                  << (param_abs->GetSymbolicShape() == nullptr ? "none" : param_abs->GetSymbolicShape()->ToString())
                  << ". V:"
                  << (param_abs->GetSymbolicValue() == nullptr ? "none" : param_abs->GetSymbolicValue()->ToString());
  }
  return build_again;
}

void SymbolEngineImpl::BuildSubgraphImpl(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  MS_EXCEPTION_IF_NULL(sub_fg);
  auto visit_cnt = ++visited_graph_[sub_fg.get()];
  MS_LOG(DEBUG) << "Build subgraph " << sub_fg->ToString() << " of node " << cnode->fullname_with_scope()
                << ". visit count: " << visit_cnt;
  bool build_again = SetParamSymbols(cnode, sub_fg, begin_input_index, visit_cnt);
  if (visit_cnt > 1) {
    if (!build_again) {
      MS_LOG(DEBUG) << "The inputs of graph " << sub_fg->ToString() << " are equal to last building, don't build again";
      return;
    }
    support_infer_ = false;
  }

  BuildNodesSymbol(sub_fg, GetCNodesOfFuncGraph(sub_fg));
  // only set the abstract for "call" node.
  if (!IsPrimitiveCNode(cnode, prim::kPrimPartial)) {
    auto out_abs = sub_fg->output()->abstract();
    MS_EXCEPTION_IF_NULL(out_abs);
    auto cnode_abs = CloneAbstractIfSymbolExists(cnode);
    MS_EXCEPTION_IF_NULL(cnode_abs);
    cnode_abs->SetSymbolicShape(out_abs->GetSymbolicShape());
    cnode_abs->SetSymbolicValue(out_abs->GetSymbolicValue());
  }
  MS_LOG(DEBUG) << "Finish to build subgraph " << sub_fg->ToString() << " of node " << cnode->fullname_with_scope()
                << ". visit count: " << visit_cnt;
}

SymbolPtr SymbolEngineImpl::BuildCNodeSymbolicShape(OperationBuilder *builder, const PrimitivePtr &prim,
                                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract,
                                                    const CNodePtr &cnode) {
  auto digital_shape = abstract->GetShape();
  MS_EXCEPTION_IF_NULL(digital_shape);
  if (common::GetEnv("MS_DEV_FORCE_BUILD_SYMBOL") != "on" && !digital_shape->IsDynamic()) {
    auto static_shape = digital_shape->BuildSymbolicShape();
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " is static shape: " << digital_shape->ToString();
    return static_shape;
  }
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildShape, builder not found.";
    return digital_shape->BuildSymbolicShape();
  }
  SymbolPtr s = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    s = builder->BuildShape(prim, inputs, abstract);
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Failed to build symbolic shape for " << cnode->fullname_with_scope() << " with inputs: " << inputs
                 << ". error msg: " << e.what();
    s = nullptr;
  }
  if (s == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildShape.";
    return digital_shape->BuildSymbolicShape();
  }
  return s;
}

SymbolPtr SymbolEngineImpl::BuildCNodeSymbolicValue(OperationBuilder *builder, const PrimitivePtr &prim,
                                                    const AbstractBasePtrList &inputs, const AbstractBasePtr &abstract,
                                                    const CNodePtr &cnode) {
  if (builder == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildValue, builder not found.";
    return BuildSymbolicValue(abstract);
  }
  SymbolPtr s = nullptr;
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    s = builder->BuildValue(prim, inputs, abstract);
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Failed to build symbolic value for " << cnode->fullname_with_scope() << " with inputs: " << inputs
                 << ". error msg: " << e.what();
    s = nullptr;
  }
  if (s == nullptr) {
    support_infer_ = false;
    MS_LOG(DEBUG) << "Node " << cnode->fullname_with_scope() << " does not support BuildValue.";
    return BuildSymbolicValue(abstract);
  }
  return s;
}

void SymbolEngineImpl::BuildCNodeSymbol(const CNodePtr &cnode) {
  PrimitivePtr prim;
  AbstractBasePtrList inputs;
  if (cnode->input(0)->isa<CNode>()) {
    if (auto iter = special_cnodes_.find(cnode); iter != special_cnodes_.end()) {
      auto ret = iter->second->ExtractInputs();
      prim = std::move(ret.first);
      inputs = std::move(ret.second);
    }
    if (prim == nullptr) {
      prim = std::make_shared<Primitive>("_SpecialCNode");
    }
  } else {
    prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      prim = std::make_shared<Primitive>("_UnsupportedCNode");
    }
    inputs = ExtractInputsAbstract(cnode);
  }
  auto abstract = CloneAbstractIfSymbolExists(cnode);
  MS_EXCEPTION_IF_NULL(abstract);
  if (HasAbstractAny(inputs, abstract)) {
    MS_LOG(DEBUG) << "The input or output of " << cnode->fullname_with_scope()
                  << " has AbstractAny, which is not supported by symbol engine. node: " << cnode->DebugString();
    return;
  }

  auto builder = OperationBuilderInfoRegistry::GetBuilder(prim->name(), emitter_.get());
  // theoretically, it's possible that both shape and value are required for a same node.
  const auto &depend_status = depend_status_map_[cnode];
  if (depend_status.value) {
    MS_LOG(DEBUG) << "Build value for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    auto sym_value = BuildCNodeSymbolicValue(builder.get(), prim, inputs, abstract, cnode);
    MS_LOG(DEBUG) << "Set value for node: " << cnode->fullname_with_scope() << ". symbol: " << sym_value->ToString();
    abstract->SetSymbolicValue(sym_value);
  }

  if (depend_status.shape) {
    MS_LOG(DEBUG) << "Build shape for node " << cnode->fullname_with_scope() << ".   " << cnode->DebugString();
    auto sym_shape = BuildCNodeSymbolicShape(builder.get(), prim, inputs, abstract, cnode);
    MS_EXCEPTION_IF_NULL(sym_shape);
    MS_LOG(DEBUG) << "Set shape for node: " << cnode->fullname_with_scope() << ". symbol: " << sym_shape->ToString();
    abstract->SetSymbolicShape(sym_shape->as_sptr<ListSymbol>());
  }
}

std::string SymbolEngineImpl::DumpText() const {
  std::ostringstream oss;
  oss << ToString() << " {\n";
  for (auto op : ops_) {
    oss << op->DumpText();
  }
  oss << "}\n";
  return oss.str();
}

void SymbolEngineImpl::GetAllNodes(const FuncGraphPtr &func_graph) {
  auto nodes = TopoSort(func_graph->output(), SuccDeeperWithAttrGraph, AlwaysInclude);
  for (auto &node : nodes) {
    if (node->isa<CNode>() && !IsPrimitiveCNode(node, prim::kPrimReturn) && node->func_graph() != nullptr) {
      (void)fg_cnodes_[node->func_graph().get()].emplace_back(node);
    }
  }
}
}  // namespace symshape
}  // namespace mindspore
