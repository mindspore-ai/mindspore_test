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
#ifndef MINDSPORE_CORE_OPS_VIEW_SPLIT_STRIDES_CALC_H
#define MINDSPORE_CORE_OPS_VIEW_SPLIT_STRIDES_CALC_H

#include <vector>
#include "view/view_strides_calculator.h"

namespace mindspore {
namespace ops {
OPS_API TensorStorageInfoPtrList SplitBasicTypeCalc(const PrimitivePtr &prim,
                                                    const mindspore::tensor::TensorPtr &input_tensor,
                                                    const int64_t &axis, const int64_t &output_num);
OPS_API TensorStorageInfoPtrList SplitCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs);
OPS_API TensorStorageInfoPtrList SplitProcess(const OldTensorInfoPtr &input_tensor, const int64_t &axis,
                                              const int64_t &output_num);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_VIEW_SPLIT_STRIDES_CALC_H
