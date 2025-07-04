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

#ifndef MINDSPORE_CORE_OPS_PADDING_H_
#define MINDSPORE_CORE_OPS_PADDING_H_
#include <memory>
#include <vector>
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class OPS_API Padding : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Padding);
  /// \brief Constructor.
  Padding() : BaseOperator(kPaddingOpName) { InitIOName({"x"}, {"y"}); }

  void Init(int64_t pad_dim_size = 8);

  int64_t get_pad_dim_size() const;

  void set_pad_dim_size(int64_t pad_dim_size);
};
OPS_API abstract::AbstractBasePtr PaddingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_PADDING_H_
