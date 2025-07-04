/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "infer/non_max_suppression.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(NonMaxSuppression, BaseOperator);
void NonMaxSuppression::set_center_point_box(const int64_t center_point_box) {
  (void)AddAttr(kCenterPointBox, api::MakeValue(center_point_box));
}
int64_t NonMaxSuppression::get_center_point_box() const {
  auto value_ptr = this->GetAttr(kCenterPointBox);
  return GetValue<int64_t>(value_ptr);
}
void NonMaxSuppression::Init(const int64_t center_point_box) { this->set_center_point_box(center_point_box); }

REGISTER_PRIMITIVE_C(kNameNonMaxSuppression, NonMaxSuppression);
}  // namespace ops
}  // namespace mindspore
