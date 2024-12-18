/**
* Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_FUNCTIONS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_FUNCTIONS_H_

#include <functional>
#include "plugin/device/ascend/kernel/internal/pyboost/add.h"
#include "plugin/device/ascend/kernel/internal/pyboost/apply_rotary_pos_emb.h"
#include "plugin/device/ascend/kernel/internal/pyboost/fast_gelu.h"
#include "plugin/device/ascend/kernel/internal/pyboost/gather.h"
#include "plugin/device/ascend/kernel/internal/pyboost/gelu.h"
#include "plugin/device/ascend/kernel/internal/pyboost/matmul.h"
#include "plugin/device/ascend/kernel/internal/pyboost/mul.h"
#include "plugin/device/ascend/kernel/internal/pyboost/quant_batch_matmul.h"
#include "plugin/device/ascend/kernel/internal/pyboost/realdiv.h"
#include "plugin/device/ascend/kernel/internal/pyboost/reshape_and_cache.h"
#include "plugin/device/ascend/kernel/internal/pyboost/sub.h"
#include "plugin/device/ascend/kernel/internal/pyboost/swiglu.h"
#include "plugin/device/ascend/kernel/internal/pyboost/swish.h"
#include "plugin/device/ascend/kernel/internal/pyboost/transpose.h"

namespace mindspore{
namespace kernel{
using AcmeFunc = std::function<void(void)>;

// TODO: use auto-generate
static std::map<std::string, AcmeFunc> acme_func_map = {{"ReshapeAndCache", ReshapeAndCacheAcmeCustomize}};
}
}

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ACME_FUNCTIONS_H_
