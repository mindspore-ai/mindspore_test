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

#ifndef MINDSPORE_CORE_OPS_MAKE_LIST_H_
#define MINDSPORE_CORE_OPS_MAKE_LIST_H_
#include "mindspore/ops/op_def/sequence_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
/// \brief MakeList op is used to pack multiple nodes into a whole, which is only used in FuncGraph.
class OPS_API MakeList : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MakeList);
  /// \brief Constructor.
  MakeList() : BaseOperator(kMakeListOpName) {}
  /// \brief Init.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAKE_LIST_H_
