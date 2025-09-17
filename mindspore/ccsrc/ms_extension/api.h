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
// Pybind11 includes
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// MS Extension includes
#include "ms_extension/common/tensor.h"
#include "ms_extension/common/tensor_utils.h"
#include "ms_extension/common/utils.h"
#include "ms_extension/pynative/pyboost_extension.h"

// Ascend files
#ifdef CUSTOM_ASCEND_OP
#include "ms_extension/ascend/aclnn/aclnn_op_runner.h"
// MindSpore core includes
#include "mindspore/core/include/utils/ms_utils.h"
#include "mindspore/core/include/ops/ops_func_impl/op_func_impl.h"
#include "mindspore/core/include/utils/convert_utils_base.h"
#include "mindspore/core/include/utils/check_convert_utils.h"
#include "mindspore/core/include/utils/ms_context.h"

// MindSpore ops includes
#include "mindspore/ops/kernel/ascend/custom/kernel_mod_impl/custom_kernel_factory.h"
#include "mindspore/ops/ops_utils/op_utils.h"

// MindSpore runtime includes
#include "mindspore/ccsrc/plugin/ascend/res_manager/mem_manager/ascend_memory_manager.h"
#include "mindspore/ccsrc/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "mindspore/ccsrc/plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "mindspore/ccsrc/runtime/hardware_abstract/device_context/device_context.h"
#include "mindspore/ccsrc/include/backend/common/ms_device_shape_transfer.h"

// Internal kernel includes
#include "mindspore/ops/kernel/ascend/internal/internal_ascend_adapter.h"

#ifdef CUSTOM_ENABLE_ATB
#include "ms_extension/ascend/atb/atb_common.h"
#endif  // CUSTOM_ENABLE_ATB
#ifdef CUSTOM_ENABLE_ASDSIP
#include "ms_extension/ascend/asdsip/asdsip_common.h"
#endif  // CUSTOM_ENABLE_ASDSIP
#endif  // CUSTOM_ASCEND_OP
#endif  // MINDSPORE_CCSRC_MS_EXTENSION_API_H_
