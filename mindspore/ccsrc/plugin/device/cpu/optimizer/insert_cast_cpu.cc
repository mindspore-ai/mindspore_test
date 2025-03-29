/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/optimizer/insert_cast_cpu.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "base/base.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "ir/anf.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/optimizer/cpu_pass_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
namespace {
constexpr unsigned int kLstmReserveIndex = 3;
std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>> GetNodeUserList(const FuncGraphPtr &graph,
                                                                         const AnfNodePtr &node) {
  auto output_node_list = std::make_shared<std::vector<std::pair<AnfNodePtr, int>>>();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    return output_node_list;
  }
  auto output_info_list = iter->second;
  (void)std::copy(output_info_list.begin(), output_info_list.end(), std::back_inserter(*output_node_list));
  return output_node_list;
}

AnfNodePtr AddAssignNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input1, const AnfNodePtr &input2) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input1);
  MS_EXCEPTION_IF_NULL(input2);
  if (input1->kernel_info() == nullptr || input2->kernel_info() == nullptr || input1->abstract() == nullptr) {
    MS_LOG(DEBUG) << "Invalid kernel info, input1:" << input1->kernel_info() << " input2:" << input2->kernel_info()
                  << " abstract:" << input1->abstract() << " for graph:" << func_graph->ToString();
    return input2;
  }
  const auto &kernel_info1 = dynamic_cast<device::KernelInfo *>(input1->kernel_info());
  const auto &kernel_info2 = dynamic_cast<device::KernelInfo *>(input2->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info1);
  MS_EXCEPTION_IF_NULL(kernel_info2);
  if (!kernel_info1->has_build_info() || !kernel_info2->has_build_info()) {
    MS_LOG(DEBUG) << "Invalid build info, input1:" << kernel_info1->has_build_info()
                  << " input2:" << kernel_info2->has_build_info() << " for graph:" << func_graph->ToString();
    return input2;
  }

  CNodePtr assign =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimAssign->name())), input1, input2});
  MS_EXCEPTION_IF_NULL(assign);
  assign->set_abstract(input1->abstract()->Clone());
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr) {
    kernel_graph->AddRefCorrespondPairs(std::make_pair(assign, 0), common::AnfAlgo::VisitKernel(input1, 0));
  }

  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({kernel_info1->select_kernel_build_info()->GetOutputFormat(0),
                           kernel_info2->select_kernel_build_info()->GetOutputFormat(0)});
  builder.SetOutputsFormat({kernel_info1->select_kernel_build_info()->GetOutputFormat(0)});
  builder.SetInputsDeviceType({kernel_info1->select_kernel_build_info()->GetOutputDeviceType(0),
                               kernel_info2->select_kernel_build_info()->GetOutputDeviceType(0)});
  builder.SetOutputsDeviceType({kernel_info1->select_kernel_build_info()->GetOutputDeviceType(0)});
  builder.SetInputsKernelObjectType({kernel_info1->select_kernel_build_info()->GetOutputKernelObjectType(0),
                                     kernel_info2->select_kernel_build_info()->GetOutputKernelObjectType(0)});
  builder.SetOutputsKernelObjectType({kernel_info1->select_kernel_build_info()->GetOutputKernelObjectType(0)});
  auto kernel_info = std::make_shared<device::KernelInfo>();
  assign->set_kernel_info(kernel_info);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), assign.get());

  auto depend_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), input1, assign});
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(assign->abstract());
  MS_LOG(DEBUG) << "Add depend node:" << depend_node->fullname_with_scope() << " for graph:" << func_graph->ToString();
  return depend_node;
}

void SyncWeightNodeWithCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const AnfNodePtr &cur_input,
                            const AnfNodePtr &cast, const std::string &format, const TypeId &device_type,
                            const TypeId &origin_type, const abstract::BaseShapePtr &origin_shape,
                            std::vector<AnfNodePtr> *make_tuple_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cast);
  MS_EXCEPTION_IF_NULL(make_tuple_inputs);
  auto first_depend_node =
    func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cast, cnode});
  MS_EXCEPTION_IF_NULL(first_depend_node);
  first_depend_node->set_abstract(cast->abstract());
  auto post_cast = AddCastOpNodeToGraph(func_graph, first_depend_node, format, device_type, origin_type, origin_shape);
  // As some cpu optimizer does not support fp16, the graph will be like:
  // para(fp16)
  // %1 = cast(para, fp32)
  // %2 = opt(%1)
  // %3 = depend(%1, %2)
  // %4 = cast(%3)
  // and the parameter should be update by %4, and should be add an assign op.
  auto assign = AddAssignNodeToGraph(func_graph, cur_input, post_cast);
  if (!common::AnfAlgo::CheckPrimitiveType(assign, prim::kPrimDepend)) {
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    kernel_graph->AddRefCorrespondPairs(std::make_pair(post_cast, 0), common::AnfAlgo::VisitKernel(cur_input, 0));
    MS_LOG(DEBUG) << "Add ref relation for cur node:" << cur_input->fullname_with_scope()
                  << " and post node:" << post_cast->fullname_with_scope() << " for graph:" << func_graph->ToString();
  }
  make_tuple_inputs->push_back(assign);
}

void InsertCast(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  size_t in_num = common::AnfAlgo::GetInputTensorNum(cnode);
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    // Do not insert cast for None input which is optional().
    if (common::AnfAlgo::IsNoneInput(cnode, input_index)) {
      continue;
    }

    auto prev_node = common::AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    auto origin_type = AnfAlgo::GetOutputDeviceDataType(prev_node.first, prev_node.second);
    if (origin_type == kTypeUnknown) {
      origin_type = common::AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    }
    auto cur_input = common::AnfAlgo::GetInputNode(cnode, input_index);
    MS_EXCEPTION_IF_NULL(cur_input);
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const abstract::BaseShapePtr origin_shape = AnfAlgo::GetOutputDetailShape(prev_node.first, prev_node.second);
    auto device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index);
    if (origin_type != device_type && origin_type != kTypeUnknown && device_type != kTypeUnknown) {
      auto cast = AddCastOpNodeToGraph(func_graph, cur_input, dev_fmt, origin_type, device_type, origin_shape);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      cnode->set_input(input_index + 1, cast);
      auto real_input = common::AnfAlgo::VisitKernel(cur_input, 0).first;
      MS_EXCEPTION_IF_NULL(real_input);
      if (common::AnfAlgo::IsUpdateParameterKernel(cnode) && real_input->isa<Parameter>() &&
          common::AnfAlgo::IsParameterWeight(real_input->cast<ParameterPtr>())) {
        SyncWeightNodeWithCast(func_graph, cnode, cur_input, cast, dev_fmt, device_type, origin_type, origin_shape,
                               &make_tuple_inputs);
      }
    }
    if (make_tuple_inputs.size() > 1) {
      auto make_tuple = func_graph->NewCNode(make_tuple_inputs);
      auto second_depend_node =
        func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), cnode, make_tuple});
      MS_EXCEPTION_IF_NULL(second_depend_node);
      second_depend_node->set_abstract(cnode->abstract());
      auto used_node_list = GetRealNodeUsedList(func_graph, cnode);
      if (used_node_list != nullptr && used_node_list->empty()) {
        used_node_list = GetNodeUserList(func_graph, cnode);
      }
      for (size_t j = 0; j < used_node_list->size(); j++) {
        auto used_node = used_node_list->at(j).first;
        MS_EXCEPTION_IF_NULL(used_node);
        if (!used_node->isa<CNode>()) {
          continue;
        }
        utils::cast<CNodePtr>(used_node)->set_input(IntToSize(used_node_list->at(j).second), second_depend_node);
      }
    }
  }
}

void InsertCastForGraphOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &func_output) {
  MS_EXCEPTION_IF_NULL(func_output);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(func_output);
  if (!func_output->isa<CNode>()) {
    return;
  }
  auto func_output_node = func_output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(func_output_node);
  for (size_t i = 0; i < input_num; i++) {
    auto input_node = common::AnfAlgo::GetInputNode(func_output_node, i);
    MS_EXCEPTION_IF_NULL(input_node);
    auto abstract = input_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (!abstract->isa<abstract::AbstractTensor>()) {
      MS_LOG(INFO) << "The " << i << "th output of graph is not a tensor type, skipping insert cast.";
      continue;
    }
    if (!input_node->isa<CNode>()) {
      MS_LOG(INFO) << "The " << i << "th output of graph is not a CNode, skipping insert cast.";
      continue;
    }
    auto kernel_node = common::AnfAlgo::VisitKernel(input_node, 0).first;
    MS_EXCEPTION_IF_NULL(kernel_node);
    auto infer_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(func_output, i);
    auto device_type = AnfAlgo::GetPrevNodeOutputDeviceDataType(func_output, i);
    const std::string dev_fmt = AnfAlgo::GetPrevNodeOutputFormat(func_output, i);
    if (infer_type != device_type && device_type != kTypeUnknown) {
      const abstract::BaseShapePtr origin_shape = AnfAlgo::GetPrevNodeOutputDetailShape(func_output_node, i);
      auto cast = AddCastOpNodeToGraph(func_graph, input_node, dev_fmt, device_type, infer_type, origin_shape);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(func_output->scope());
      func_output_node->set_input(i + 1, cast);
      auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
      if (kernel_graph != nullptr) {
        MS_LOG(INFO) << "Replace internal output from:" << kernel_node->DebugString() << " to:" << cast->DebugString()
                     << " for graph:" << kernel_graph->ToString();
        kernel_graph->ReplaceInternalOutput(kernel_node, cast);
      }
    }
  }
}
}  // namespace

bool InsertCastCPU::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
      CNodePtr cnode = node->cast<CNodePtr>();
      InsertCast(func_graph, cnode);
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto func_output = func_graph->output();
  InsertCastForGraphOutput(func_graph, func_output);
  return true;
}
}  // namespace opt
}  // namespace mindspore
