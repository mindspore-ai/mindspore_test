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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/gmm_common_utils.h"
#include <set>
#include "mindspore/core/include/utils/value_utils.h"

namespace mindspore::prim {
void CheckNumOfSequenceTensor(const AbstractBasePtr &abs, size_t expect_num, const std::string &op_name,
                              const std::string &arg_name) {
  auto base_shape = abs->GetShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto is_dyn_seq = base_shape->isa<abstract::DynamicSequenceShape>();
  if (is_dyn_seq) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the tensor num in '" << arg_name
                             << "' should not be dynamic, which is not supported.";
  }
  auto sequence_shape = base_shape->cast<abstract::TupleShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  auto num = sequence_shape->size();
  if (num != expect_num) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the tensor num in '" << arg_name << "' should be " << expect_num
                             << ", but got " << num;
  }
}

void CheckGroupTypeValue(const AbstractBasePtr &group_type_abs, const std::string &op_name) {
  auto group_type_opt = GetScalarValue<int64_t>(group_type_abs->GetValue());
  if (MS_UNLIKELY(!group_type_opt.has_value())) {
    MS_EXCEPTION(RuntimeError) << "For '" << op_name << "', group_type should not be dynamic.";
  }
  auto group_type = group_type_opt.value();
  static std::set<int64_t> valid_group_type_list{0, 2};
  if (MS_UNLIKELY(valid_group_type_list.find(group_type) == valid_group_type_list.end())) {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', group_type should be 0 or 2, but got " << group_type;
  }
}
}  // namespace mindspore::prim
