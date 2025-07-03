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
#include <algorithm>
#include <memory>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "view/split_with_size_strides_calc.h"
#include "view/split_with_size_view_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList SplitWithSizeViewBasicTypeCalc(const PrimitivePtr &prim,
                                                        const mindspore::tensor::TensorPtr &input_tensor,
                                                        const std::vector<int64_t> &split_size, const int64_t &dim) {
  return SplitWithSizeBasicTypeCalc(prim, input_tensor, split_size, dim);
}
TensorStorageInfoPtrList SplitWithSizeViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  return SplitWithSizeCalc(prim, inputs);
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(SplitWithSizeView, SplitWithSizeViewCalc);
}  // namespace mindspore::ops
