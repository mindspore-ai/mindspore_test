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

#include "view/expand_dims_view_strides_calc.h"
#include <memory>
#include "view/expand_dims_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList ExpandDimsViewBasicTypeCalc(const PrimitivePtr &prim,
                                                     const mindspore::tensor::TensorPtr &input_tensor,
                                                     const int64_t &dim) {
  return ExpandDimsBasicTypeCalc(prim, input_tensor, dim);
}

TensorStorageInfoPtrList ExpandDimsViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  return ExpandDimsCalc(prim, inputs);
}

REG_VIEW_STRIDES_CALC_FUN(ExpandDimsView, ExpandDimsViewCalc);
}  // namespace mindspore::ops
