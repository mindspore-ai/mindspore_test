/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "infer/hashtable_lookup.h"

#include "mindapi/helper.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(HashtableLookup, BaseOperator);
REGISTER_PRIMITIVE_C(kNameHashtableLookup, HashtableLookup);
}  // namespace ops
}  // namespace mindspore
