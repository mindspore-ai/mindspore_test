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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SPECTRAL_OP_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SPECTRAL_OP_DECLARE_H_

#include "plugin/res_manager/ascend/op_adapter/custom_op_proto/cust_spectral_ops.h"
#include "plugin/res_manager/ascend/op_adapter/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_CUST_OP_ADAPTER(BlackmanWindow)
DECLARE_CUST_OP_USE_OUTPUT(BlackmanWindow)

DECLARE_CUST_OP_ADAPTER(BartlettWindow)
DECLARE_CUST_OP_USE_OUTPUT(BartlettWindow)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SPECTRAL_OP_DECLARE_H_
