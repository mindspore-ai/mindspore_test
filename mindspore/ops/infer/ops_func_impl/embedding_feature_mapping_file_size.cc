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

#include "infer/ops_func_impl/embedding_feature_mapping_file_size.h"

#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
int32_t EmbeddingFeatureMappingFileSizeFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const auto &table_name_shape = input_args[table_name_idx_]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(table_name_shape))) {
    return OP_CHECK_RETRY;
  }

  auto table_num = SizeOf(table_name_shape);
  auto embedding_dim_opt = GetArrayValue<int64_t>(input_args[table_name_idx_ + kIndex1]);
  if (MS_UNLIKELY(!embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  auto embedding_dim = embedding_dim_opt.value();
  MS_CHECK_VALUE(embedding_dim.size() == table_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("num of embedding_dim", embedding_dim.size(), kEqual,
                                                             table_num, primitive));
  for (size_t i = 0; i < table_num; i++) {
    if (MS_UNLIKELY(embedding_dim.IsValueUnknown(i))) {
      return OP_CHECK_RETRY;
    }
    MS_CHECK_VALUE(embedding_dim[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", embedding_dim[i],
                                                                                     kGreaterThan, 0, primitive));
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
