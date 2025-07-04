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

#include "backend/common/pass/split_inputs_for_reduce_scatter.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace opt {
std::vector<AnfNodePtr> SplitInputsForReduceScatter::InsertSplitForInput(const FuncGraphPtr &func_graph,
                                                                         const CNodePtr &node,
                                                                         int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t inputs_size = common::AnfAlgo::GetInputTensorNum(node);
  std::vector<AnfNodePtr> split_outputs;
  size_t rank_size_t = LongToSize(rank_size);
  if (rank_size == 0) {
    MS_LOG(EXCEPTION) << "The rank size can not be zero.";
  }
  for (size_t i = 0; i < inputs_size; i++) {
    std::vector<AnfNodePtr> split_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimSplit->name()))};
    split_inputs.push_back(common::AnfAlgo::GetInputNode(node, i));

    // Add axis and output number nodes as the second and third input of Split.
    auto axis_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(kIndex0), kInt64);
    ValueNodePtr axis_node = std::make_shared<ValueNode>(MakeValue<int64_t>(kIndex0));
    axis_node->set_abstract(axis_abs);
    axis_node = func_graph->cast<KernelGraphPtr>()->NewValueNode(axis_node);
    func_graph->cast<KernelGraphPtr>()->AddValueNodeToGraph(axis_node);
    split_inputs.push_back(axis_node);

    auto output_num_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(rank_size_t), kInt64);
    ValueNodePtr output_num_node = std::make_shared<ValueNode>(MakeValue<int64_t>(rank_size_t));
    output_num_node->set_abstract(output_num_abs);
    output_num_node = func_graph->cast<KernelGraphPtr>()->NewValueNode(output_num_node);
    func_graph->cast<KernelGraphPtr>()->AddValueNodeToGraph(output_num_node);
    split_inputs.push_back(output_num_node);

    // Create Split node.
    auto split = func_graph->NewCNode(split_inputs);
    MS_EXCEPTION_IF_NULL(split);
    std::vector<TypeId> dtypes(rank_size, common::AnfAlgo::GetPrevNodeOutputInferDataType(node, i));

    std::vector<int> size_splits;
    auto output_node_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, i);
    output_node_shape[0] /= rank_size;
    std::vector<ShapeVector> shapes(rank_size_t, output_node_shape);
    common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());

    for (size_t j = 0; j < rank_size_t; j++) {
      size_splits.push_back(output_node_shape[0]);
    }
    common::AnfAlgo::SetNodeAttr("split_dim", MakeValue(0L), split);
    common::AnfAlgo::SetNodeAttr("num_split", MakeValue(rank_size), split);
    common::AnfAlgo::SetNodeAttr("size_splits", MakeValue(size_splits), split);
    std::vector<AnfNodePtr> new_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, split, rank_size_t, &new_outputs);
    for (size_t j = 0; j < new_outputs.size(); j++) {
      split_outputs.push_back(new_outputs[j]);
    }
  }
  return split_outputs;
}

AnfNodePtr SplitInputsForReduceScatter::RearrangeInputsForReduceScatter(const FuncGraphPtr &func_graph,
                                                                        const AnfNodePtr &node,
                                                                        const std::vector<AnfNodePtr> &inputs,
                                                                        int64_t rank_size) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t inputs_size = common::AnfAlgo::GetInputTensorNum(node);
  std::vector<AnfNodePtr> reduce_scatter_inputs{
    NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceScatter->name()))};
  size_t rank_size_t = LongToSize(rank_size);
  for (size_t i = 0; i < rank_size_t; i++) {
    for (size_t j = 0, idx = i; j < inputs_size; j++, idx += rank_size_t) {
      reduce_scatter_inputs.push_back(inputs[idx]);
    }
  }
  auto reduce_scatter = NewCNode(reduce_scatter_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(reduce_scatter);
  reduce_scatter->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttrs(node, reduce_scatter);
  common::AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(static_cast<int64_t>(1)), reduce_scatter);
  return reduce_scatter;
}

const BaseRef SplitInputsForReduceScatter::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto prim = std::make_shared<Primitive>(kReduceScatterOpName);
  return VectorRef({prim, Xs});
}

const AnfNodePtr SplitInputsForReduceScatter::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                      const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (common::AnfAlgo::GetInputTensorNum(node) == 1) {
    common::AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(0L), node);
    return nullptr;
  }
  if (!common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) || !common::AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    return nullptr;
  }
  auto fusion = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  if (fusion <= 0) {
    return nullptr;
  }
  if (common::AnfAlgo::HasNodeAttr("Fused", cnode)) {
    return nullptr;
  }

  common::AnfAlgo::SetNodeAttr("Fused", MakeValue(true), node);
  auto rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrRankSize);
  std::vector<AnfNodePtr> split_outputs = InsertSplitForInput(func_graph, cnode, rank_size);
  return RearrangeInputsForReduceScatter(func_graph, node, split_outputs, rank_size);
}
}  // namespace opt
}  // namespace mindspore
