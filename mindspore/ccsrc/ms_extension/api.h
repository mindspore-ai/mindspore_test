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
#ifndef MINDSPORE_CCSRC_MS_EXTENSION_API_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_API_H_
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "mindspore/ccsrc/pynative/grad/function.h"
#include "ms_extension/common/tensor.h"
#include "ms_extension/common/tensor_utils.h"
#include "ms_extension/pynative/pyboost_extension.h"

// ascend files
#ifdef CUSTOM_ASCEND_OP
#include "mindspore/ops/kernel/ascend/pyboost/customize/custom_launch_aclnn.h"
#ifdef CUSTOM_ENABLE_ATB
#include "ms_extension/ascend/atb/atb_common.h"
#endif  // CUSTOM_ENABLE_ATB
#endif  // CUSTOM_ASCEND_OP
#endif  // MINDSPORE_CCSRC_MS_EXTENSION_API_H_
