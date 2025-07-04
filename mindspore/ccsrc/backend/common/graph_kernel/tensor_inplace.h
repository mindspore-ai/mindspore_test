/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TENSOR_INPLACE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TENSOR_INPLACE_H_

#include <string>
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/inplace_assign_builder.h"
namespace mindspore::graphkernel {
/**
 * @brief Inplace output tensor of a graph kernel to its input tensor if requirements satisfied
 * @example
 * main(x, y, z){
 *   %1 = Op1(x, y)
 *   %2 = subgraph(%1, z)
 *   %3 = Op2(%2)
 *   return %3
 * }
 * ----
 * subgraph(a, b){
 *   %1 = Add(a, b)
 *   %2 = Exp(%1)
 *   %3 = Mul(%2, b)
 *   return %3
 * }
 *  --------------->
 * main(x, y, z){
 *   %1 = Op1(x, y)
 *   %2 = subgraph(%1, z)
 *   %3 = Depend(%1, %2)
 *   %4 = Op2(%3)
 *   return %4
 * }
 * ----
 * subgraph(a, b){
 *   %1 = Add(a, b)
 *   %2 = Exp(%1)
 *   %3 = Mul(%2, b)
 *   %4 = Assign(a, %3)
 *   return %4
 * }
 */
class TensorInplace : public InplaceAssignBuilder {
 public:
  explicit TensorInplace(const std::string &name = "tensor_inplace") : InplaceAssignBuilder(name) {}
  ~TensorInplace() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  void SetTargetAttrs(const CNodePtr &cnode) override {
    SetNodeAttrSafely("enable_auto_inplace", MakeValue(true), cnode);
  }
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_TENSOR_INPLACE_H_
