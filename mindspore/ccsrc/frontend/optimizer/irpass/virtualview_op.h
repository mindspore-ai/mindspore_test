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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_

#include <unordered_map>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"

namespace mindspore {
namespace opt {
namespace irpass {
class VirtualViewInsertProcesser {
 public:
  VirtualViewInsertProcesser(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager)
      : func_graph_(func_graph), manager_(manager) {}
  virtual ~VirtualViewInsertProcesser() = default;

  void Run();

 private:
  using ViewDependenceMap = std::unordered_map<AnfNodePtr, AnfNodePtr>;
  using ViewModificationMap = std::unordered_map<AnfNodePtr, std::unordered_map<AnfNodePtr, bool>>;
  using ViewChainMap = std::unordered_map<AnfNodePtr, CNodePtrList>;

  static constexpr auto kIsVirtualViewOp = "is_virtual_view_op";
  static constexpr auto kOriginalViewOp = "view_op";

  AnfNodePtr CreateVirtualViewNode(const CNodePtr &view_output, AnfNodePtr *last_umonad);
  void ResetViewModificationStatus(const AnfNodePtr &view_output);
  void VirtualViewInsertAction(const CNodePtr &cnode, const CNodePtr &view_cnode);
  void UpdateViewModificationStatus(const AnfNodePtr &input_node);
  void UpdateViewChains(const CNodePtr &view_cnode);
  AnfNodePtr FindViewRootNode(const CNodePtr &view_cnode);
  void ProcessViewNode(const CNodePtr &cnode);
  void ProcessInplaceNode(const CNodePtr &cnode);
  void CheckAndInsertVirtualViewOp(const CNodePtr &cnode);
  void ChangeVirtualViewInputInner();
  bool IsVirtualViewCNode(const AnfNodePtr &node);
  void DoVirtualViewInputReplace();

  FuncGraphPtr func_graph_;
  FuncGraphManagerPtr manager_;
  // m = View(y), n = View(m) -> {m: y, n: y}
  ViewDependenceMap view_dependencies_;
  // m = View(y), n = View(m), Inplace(y) -> {y: {m: true, n: true}}
  ViewModificationMap view_modifications_;
  // m = View(y), n = View(m) -> {m: [m], n: [m, n]}
  ViewChainMap view_chains_;
};

void VirtualViewInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt);

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIRTUALVIEW_INSERT_H_
