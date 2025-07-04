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

#include "infer/cxx_api/flash_attention.h"
#include "mindapi/helper.h"
#include "ops/primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace ops {
void FlashAttention::Init() const {}
MIND_API_OPERATOR_IMPL(FlashAttention, BaseOperator);
REGISTER_PRIMITIVE_C(kNameFlashAttention, FlashAttention);
}  // namespace ops
}  // namespace mindspore
