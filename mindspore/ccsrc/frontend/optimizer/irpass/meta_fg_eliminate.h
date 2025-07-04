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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_ELIMINATE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_ELIMINATE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "frontend/optimizer/irpass/vmap_eliminate.h"
#include "frontend/optimizer/irpass/taylor_eliminate.h"
#include "frontend/optimizer/irpass/shard_eliminate.h"
#include "frontend/optimizer/irpass/meta_fg_prim_eliminate.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ExpandMetaFg {
 public:
  ExpandMetaFg() {
    // Register derived class of `ExpandMetaFgPrim` to `ExpandMetaFg`, note that corresponding modifications
    // need to be made in the `manager.cc` to support `ExpandMetaFgPrim::CheckIfEmbedMetaFgPrim`, analogous
    // to the implementation of `kPrimVmap`.
    (void)expand_meta_fg_list_.emplace_back("J", std::make_shared<ExpandJPrim>());
    (void)expand_meta_fg_list_.emplace_back("vmap", std::make_shared<ExpandVmapPrim>());
    (void)expand_meta_fg_list_.emplace_back("taylor", std::make_shared<ExpandTaylorPrim>());
  }
  virtual ~ExpandMetaFg() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);

 private:
  std::vector<std::pair<std::string, ExpandMetaFGPrimPtr>> expand_meta_fg_list_;
};
class ExpandMetaShardFg {
 public:
  ExpandMetaShardFg() {
    // Register derived class of `ExpandShardPrim` to `ExpandShardFg`
    (void)expand_meta_shard_fg_list_.emplace_back("shard", std::make_shared<ExpandShardPrim>());
  }
  virtual ~ExpandMetaShardFg() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);

 private:
  std::vector<std::pair<std::string, ExpandMetaFGPrimPtr>> expand_meta_shard_fg_list_;
};

class ExpandMetaAddAttrFg {
 public:
  ExpandMetaAddAttrFg() {
    (void)expand_meta_addattr_fg_list_.emplace_back("addattr", std::make_shared<ExpandAddAttrPrim>());
  }
  virtual ~ExpandMetaAddAttrFg() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);

 private:
  std::vector<std::pair<std::string, ExpandMetaFGPrimPtr>> expand_meta_addattr_fg_list_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_META_FG_ELIMINATE_H_
