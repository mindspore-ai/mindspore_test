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
#include "include/common/symbol_engine/utils.h"
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

namespace mindspore {
namespace symshape {
AbstractBasePtrList ExtractInputsAbstract(const CNodePtr &cnode) {
  AbstractBasePtrList abs_list;
  abs_list.reserve(cnode->size());
  (void)std::transform(cnode->inputs().cbegin() + 1, cnode->inputs().cend(), std::back_inserter(abs_list),
                       [](const AnfNodePtr &node) {
                         MS_EXCEPTION_IF_NULL(node);
                         auto abs = node->abstract();
                         if (abs == nullptr) {
                           if (node->isa<ValueNode>()) {
                             auto vnode = node->cast_ptr<ValueNode>();
                             MS_EXCEPTION_IF_NULL(vnode);
                             MS_EXCEPTION_IF_NULL(vnode->value());
                             abs = vnode->value()->ToAbstract();
                             MS_EXCEPTION_IF_NULL(abs);
                             node->set_abstract(abs);
                             MS_LOG(DEBUG) << "Set new abstract for input node " << node->DebugString();
                           } else {
                             // Do not raise exception here, this input may not be used by operation.
                             MS_LOG(INFO) << "The input " << node->DebugString() << " has null abstract.";
                           }
                         }
                         return abs;
                       });
  return abs_list;
}

bool HasAbstractAny(const AbstractBasePtrList &inputs, const AbstractBasePtr &output) {
  return output->isa<abstract::AbstractAny>() ||
         std::any_of(inputs.begin(), inputs.end(),
                     [](const AbstractBasePtr &abs) { return abs->isa<abstract::AbstractAny>(); });
}

AbstractBasePtr CloneAbstractIfSymbolExists(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return nullptr;
  }
  if (abs->GetSymbolicShape() == nullptr && abs->GetSymbolicValue() == nullptr) {
    return abs;
  }
  // some abstract does not support clone
  if (abs->isa<abstract::AbstractFuncUnion>()) {
    return abs;
  }
  try {
    MS_LOG_TRY_CATCH_SCOPE;
    auto new_abs = abs->Clone();
    MS_EXCEPTION_IF_NULL(new_abs);
    new_abs->SetSymbolicShape(nullptr);
    new_abs->SetSymbolicValue(nullptr);
    return new_abs;
  } catch (std::exception &e) {
    if (IS_OUTPUT_ON(MsLogLevel::kDebug)) {
      std::string sym_shape = abs->GetSymbolicShape() == nullptr ? "" : abs->GetSymbolicShape()->ToString();
      std::string sym_value = abs->GetSymbolicValue() == nullptr ? "" : abs->GetSymbolicValue()->ToString();
      MS_LOG(DEBUG) << "The abstract has symbol (S:" << sym_shape << ", V:" << sym_value
                    << ") but cannot be cloned. abstract: " << abs->ToString() << ", msg:" << e.what();
    }
  }
  return abs;
}

void CleanSymbols(const FuncGraphPtr &func_graph) {
  std::set<AbstractBasePtr> params_abs;
  for (auto &param : func_graph->parameters()) {
    (void)params_abs.insert(param->abstract());
  }
  auto nodes = TopoSort(func_graph->get_return(), SuccDeeperWithAttrGraph, AlwaysInclude);
  for (auto &node : nodes) {
    auto abs = node->abstract();
    if (params_abs.find(abs) != params_abs.end()) {
      // do not clean the parameters' symbol
      continue;
    }
    if (abs != nullptr) {
      abs->SetSymbolicShape(nullptr);
      abs->SetSymbolicValue(nullptr);
    }
    auto fg = node->func_graph();
    if (fg != nullptr) {
      fg->set_symbol_engine(nullptr);
    }
  }
}
}  // namespace symshape
}  // namespace mindspore
