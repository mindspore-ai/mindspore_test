/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/utils/allocator.h"
#include <vector>
#include "pipeline/jit/pi/graph_capture/cfg.h"

namespace mindspore {
namespace pijit {
Allocator::~Allocator() {
  for (Instr *i : instr_pool_) {
    delete i;
  }
  instr_pool_.clear();
  for (AbstractNode *i : node_pool_) {
    delete i;
  }
  node_pool_.clear();
}

InstrNode *Allocator::NewInstrNode(int op, int arg) { return NewNode<InstrNode>(op, arg); }

ValueNode *Allocator::NewValueNode(AObject *a, int b, int c, const std::vector<ValueNode *> &d) {
  return NewNode<ValueNode>(a, b, c, d);
}
}  // namespace pijit
}  // namespace mindspore
