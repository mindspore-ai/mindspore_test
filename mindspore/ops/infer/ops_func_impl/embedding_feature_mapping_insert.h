/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_EMBEDDING_FEATURE_MAPPING_INSERT_H_
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_EMBEDDING_FEATURE_MAPPING_INSERT_H_

#include <tuple>
#include <vector>
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API EmbeddingFeatureMappingInsertFuncImpl : public OpFuncImpl {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

  int32_t CheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;

 protected:
  std::tuple<int32_t, size_t, std::vector<int64_t>> CommonCheck(const PrimitivePtr &primitive,
                                                                const std::vector<AbstractBasePtr> &input_args) const;

  virtual void SetDynInputSizes(const PrimitivePtr &primitive, int64_t table_num) const;

  virtual int32_t SpecifiedCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                 size_t table_num, const std::vector<int64_t> &feature_size) const;

  size_t table_name_idx_{0};
  size_t feature_id_idx_{2};
  size_t other_arg_num_{2};
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_EMBEDDING_FEATURE_MAPPING_INSERT_H_
