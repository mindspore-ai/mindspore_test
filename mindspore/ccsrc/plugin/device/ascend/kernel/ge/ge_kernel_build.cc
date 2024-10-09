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
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "plugin/device/ascend/kernel/ge/ge_kernel_build.h"
#include "plugin/device/ascend/kernel/ge/ge_kernel_mod.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/framework_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/trace_base.h"
#include "op_def/framework_ops.h"

namespace mindspore {
namespace kernel {
namespace {
static const char kAlreadyCompile[] = "AlreadyCompile";
}  // namespace

KernelModPtr GeOpBuild(const AnfNodePtr &anf_node, device::ascend::GeGraphExecutor *graph_executor) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(graph_executor);

  if (!common::AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimGEGraphOp)) {
    MS_LOG(EXCEPTION) << "Current node must be GEGraphOp! but got " << anf_node->DebugString();
  }

  auto kernel_mod_ptr = std::make_shared<GeKernelMod>();
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(primitive);
  if (!std::static_pointer_cast<KernelMod>(kernel_mod_ptr)
         ->Init(primitive, input_kernel_tensors, output_kernel_tensors)) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#Initialize ge kernel op[" << anf_node->fullname_with_scope() << "] failed."
      << trace::DumpSourceLines(anf_node);
  }

  if (kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
    MS_LOG_WITH_NODE(EXCEPTION, anf_node)
      << "#dmsg#Kernel build failed:#dmsg#hostapi kernel op[" << anf_node->fullname_with_scope() << "] Resize failed.";
  }

  auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(anf_node, kAttrKernelGraph);
  MS_LOG(INFO) << "GeOpBuild, node name: " << anf_node->fullname_with_scope() << ", " << inline_subgraph->ToString();
  MS_EXCEPTION_IF_NULL(inline_subgraph);
  if (AnfAlgo::IsNoRealKernelGraph(inline_subgraph)) {
    kernel_mod_ptr->set_skip_run(true);
    return kernel_mod_ptr;
  }
  kernel_mod_ptr->set_executor(graph_executor);
  kernel_mod_ptr->set_graph(inline_subgraph);
  kernel_mod_ptr->set_kernel(anf_node);

  if (!inline_subgraph->has_flag(kAlreadyCompile)) {
    graph_executor->CompileGraphForKernel(inline_subgraph);
    // Initialize GeTensor here for save time in RunGraph
    graph_executor->InitGraphInfo(inline_subgraph);
    inline_subgraph->set_flag(kAlreadyCompile, true);
  }

  std::vector<size_t> workspace_list;
  auto insert_func = [&workspace_list](size_t mem) {
    if (mem != 0) {
      (void)workspace_list.emplace_back(mem);
    }
  };
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (kernel_mod_ptr->Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "#dmsg#Kernel build failed:#dmsg#ge kernel op["
                                         << cnode->fullname_with_scope() << "] Resize failed.";
    }
    const auto &key = device::ascend::GetGraphName(inline_subgraph);
    auto dynamic_mem = graph_executor->GetGraphWorkSpaceMemory(key);
    insert_func(dynamic_mem);
    kernel_mod_ptr->SetWorkspaceSizeList(workspace_list);
  }

  kernel_mod_ptr->set_io_indexes(graph_executor->GetGraphRefIndexes(inline_subgraph));

  return kernel_mod_ptr;
}
}  // namespace kernel
}  // namespace mindspore
