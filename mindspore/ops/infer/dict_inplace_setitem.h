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

#ifndef MINDSPORE_CORE_OPS_DICT_INPLACE_SETITEM_H
#define MINDSPORE_CORE_OPS_DICT_INPLACE_SETITEM_H

#include "mindspore/ops/op_def/sequence_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
/// \brief Dict inplace setitem operation 'dict[index] = target'.
class OPS_API DictInplaceSetItem : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DictInplaceSetItem);
  /// \brief Constructor.
  DictInplaceSetItem() : BaseOperator("dict_inplace_setitem") {
    InitIOName({"dict", "index", "target"}, {"output_data"});
  }
  /// \brief Init function.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_DICT_INPLACE_SETITEM_H
