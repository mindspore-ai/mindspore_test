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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_TRANSPOSE_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_TRANSPOSE_H_

#include <set>
#include "infer/ops_func_impl/grouped_matmul_v4.h"

namespace mindspore {
namespace ops {
class OPS_API GroupedMatmulV4TransposeFuncImpl final : public GroupedMatmulV4FuncImpl {
 public:
  GroupedMatmulV4TransposeFuncImpl() {
    idxes_.x = 0;
    idxes_.weight = 1;
    idxes_.split_item_offset = -6;
    idxes_.group_type_offset = -5;
    idxes_.transpose_a_offset = -2;
    idxes_.transpose_b_offset = -1;
  }
  ~GroupedMatmulV4TransposeFuncImpl() = default;

 protected:
  bool GetTransposeValue(const InferInfoPtrList &input_infos, int64_t transpose_index) const override;

  int32_t PrivateCheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                 int64_t group_type) const override;

 private:
  int64_t group_list_idx_ = 8;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_GROUPED_MATMUL_V4_TRANSPOSE_H_
