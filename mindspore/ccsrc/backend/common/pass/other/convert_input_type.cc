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

#include "mindspore/ccsrc/backend/common/pass/other/convert_input_type.h"

#include <functional>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ccsrc/backend/common/pass/common/get_value_helper.h"

namespace mindspore {
namespace opt {
namespace {
const std::map<std::string, std::vector<std::pair<size_t, std::pair<TypeId, TypeId>>>> kConvertInputTypeMap = {
  {prim::kPrimRange->name(),
   {{0, {kNumberTypeFloat64, kNumberTypeFloat32}},
    {1, {kNumberTypeFloat64, kNumberTypeFloat32}},
    {2, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimApplyCamePart1->name(), {{1, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimApplyCamePart2->name(), {{6, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimApplyCamePart3->name(),
   {{2, {kNumberTypeFloat64, kNumberTypeFloat32}},
    {3, {kNumberTypeFloat64, kNumberTypeFloat32}},
    {4, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimApplyCamePart4->name(), {{6, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimFFTFreq->name(), {{1, {kNumberTypeFloat64, kNumberTypeFloat32}}}},
  {prim::kPrimRFFTFreq->name(), {{1, {kNumberTypeFloat64, kNumberTypeFloat32}}}}};

template <typename S, typename T>
const AnfNodePtr GetTransConstInputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) {
  auto ori_value = GetNodeScalarValue<S>(input_node);
  auto tar_value = static_cast<T>(ori_value);
  auto new_input_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(tar_value));
  return new_input_node;
}

using TransConstFunc = std::function<const AnfNodePtr(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node)>;
const std::map<std::pair<TypeId, TypeId>, TransConstFunc> TransConstScalarFuncMap{
  {std::make_pair(kNumberTypeFloat64, kNumberTypeFloat32), GetTransConstInputNode<double, float>}};

const AnfNodePtr GetTransDynInputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                      const AnfNodePtr &input_node, TypeId dst_type) {
  auto scalar_cast_prim = std::make_shared<Primitive>(prim::kPrimScalarCast->name());
  MS_EXCEPTION_IF_NULL(scalar_cast_prim);

  AnfNodePtrList inputs = {NewValueNode(scalar_cast_prim), input_node};
  auto dst_type_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue(static_cast<int64_t>(dst_type)));
  inputs.push_back(dst_type_node);
  auto scalar_cast_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(scalar_cast_node);

  std::vector<AnfNodePtr> scalar_cast_inputs{input_node, dst_type_node};
  auto scalar_cast_abs = InferAbstract(scalar_cast_prim, scalar_cast_inputs);
  MS_EXCEPTION_IF_NULL(scalar_cast_abs);
  scalar_cast_node->set_abstract(scalar_cast_abs);
  scalar_cast_node->set_scope(node->scope());
  return scalar_cast_node;
}

void ConvertInputTypeForCNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode, size_t input_idx,
                              const std::pair<TypeId, TypeId> &type_pair) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(input_idx + kIndex1);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_abstract = input_node->abstract();
  MS_EXCEPTION_IF_NULL(input_abstract);
  if (!input_abstract->isa<abstract::AbstractScalar>()) {
    return;
  }
  auto input_type_id = input_abstract->GetType()->type_id();
  if (input_type_id != type_pair.first) {
    return;
  }
  AnfNodePtr cast_node{nullptr};
  if (input_node->isa<ValueNode>()) {
    auto it = TransConstScalarFuncMap.find(type_pair);
    if (it == TransConstScalarFuncMap.end()) {
      return;
    }
    cast_node = it->second(func_graph, input_node);
  } else {
    cast_node = GetTransDynInputNode(func_graph, cnode, input_node, type_pair.second);
  }
  common::AnfAlgo::SetNodeInput(cnode, cast_node, input_idx);
}
}  // namespace

const AnfNodePtr ConvertInputType::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }

  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto it = kConvertInputTypeMap.find(op_name);
  if (it == kConvertInputTypeMap.end()) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &convert_inputs = it->second;
  for (auto &input_info : convert_inputs) {
    auto idx = input_info.first;
    const auto &type_pair = input_info.second;
    ConvertInputTypeForCNode(graph, cnode, idx, type_pair);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
