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

#include "infer/ops_func_impl/custom.h"
#include <algorithm>
#include <string>

#if !defined(_WIN32) && !defined(_WIN64) && !defined(BUILD_LITE) && !defined(__APPLE__)
#include <dlfcn.h>
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "pybind_api/ir/primitive_py.h"
#endif

#include "include/common/utils/utils.h"
#include "utils/check_convert_utils.h"
#include "utils/file_utils.h"
#include "utils/custom_aot_extra.h"
#include "mindspore/core/include/abstract/abstract_value.h"

namespace mindspore::ops {

InferShapeCallback CustomFuncImpl::infer_shape_func_ = nullptr;
InferTypeCallback CustomFuncImpl::infer_type_func_ = nullptr;

BaseShapePtr CustomFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  if (infer_shape_func_) {
    return infer_shape_func_(primitive, input_args);
  }
  MS_LOG(EXCEPTION) << "Infer shape func is nullptr";
}

TypePtr CustomFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  if (infer_type_func_) {
    return infer_type_func_(primitive, input_args);
  }
  MS_LOG(EXCEPTION) << "Infer type func is nullptr";
}
}  // namespace mindspore::ops
