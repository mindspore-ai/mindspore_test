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
#include "infer/ops_func_impl/embedding_feature_mapping_export.h"

#include <vector>
#include <set>
#include <string>

#include "ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void EmbeddingFeatureMappingExportFuncImpl::SetDynInputSizes(const PrimitivePtr &primitive, int64_t table_num) const {
  (void)primitive->AddAttr("dyn_input_sizes", MakeValue(std::vector<int64_t>{-1, -1, -1, -1, table_num, table_num}));
}

int32_t EmbeddingFeatureMappingExportFuncImpl::SpecifiedCheck(const PrimitivePtr &primitive,
                                                              const std::vector<AbstractBasePtr> &input_args,
                                                              size_t table_num,
                                                              const std::vector<int64_t> &feature_size) const {
  auto embedding_dim_opt = GetArrayValue<int64_t>(input_args[embedding_dim_idx_]);
  if (MS_UNLIKELY(!embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }

  const auto &embedding_dim = embedding_dim_opt.value();
  MS_CHECK_VALUE(embedding_dim.size() == table_num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", SizeToLong(embedding_dim.size()), kEqual,
                                                             SizeToLong(table_num), primitive));
  for (size_t i = 0; i < embedding_dim.size(); i++) {
    if (MS_UNLIKELY(embedding_dim.IsValueUnknown(i))) {
      return OP_CHECK_RETRY;
    }
    MS_CHECK_VALUE(embedding_dim[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", embedding_dim[i],
                                                                                     kGreaterThan, 0, primitive));
  }

  const auto &values_shape = input_args[values_idx_]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(embedding_dim.HasUnknownValue() || IsDynamic(values_shape))) {
    return OP_CHECK_RETRY;
  }

  auto values_num = SizeToLong(SizeOf(values_shape));
  int64_t values_num_expected{0};
  for (size_t i = 0; i < table_num; i++) {
    values_num_expected += embedding_dim[i] * feature_size[i];
  }
  if (MS_UNLIKELY(values_num > 1 && values_num_expected != values_num)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", values' num doesn't match with embedding_dim and feature_size.";
  }

  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
