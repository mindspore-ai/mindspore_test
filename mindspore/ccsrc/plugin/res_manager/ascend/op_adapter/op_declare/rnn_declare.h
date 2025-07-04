/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RNN_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RNN_DECLARE_H_

#include "plugin/res_manager/ascend/op_adapter/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(BasicLSTMCell)
DECLARE_OP_USE_OUTPUT(BasicLSTMCell)

DECLARE_OP_ADAPTER(BasicLSTMCellInputGrad)
DECLARE_OP_USE_OUTPUT(BasicLSTMCellInputGrad)

DECLARE_OP_ADAPTER(BasicLSTMCellWeightGrad)
DECLARE_OP_USE_OUTPUT(BasicLSTMCellWeightGrad)

DECLARE_OP_ADAPTER(BasicLSTMCellCStateGrad)
DECLARE_OP_USE_OUTPUT(BasicLSTMCellCStateGrad)

DECLARE_OP_ADAPTER(LSTMInputGrad)
DECLARE_OP_USE_OUTPUT(LSTMInputGrad)

DECLARE_OP_ADAPTER(DynamicRNN)
DECLARE_OP_USE_OUTPUT(DynamicRNN)

DECLARE_OP_ADAPTER(DynamicRNNGrad)
DECLARE_OP_USE_OUTPUT(DynamicRNNGrad)

DECLARE_OP_ADAPTER(DynamicGRUV2)
DECLARE_OP_USE_OUTPUT(DynamicGRUV2)

DECLARE_OP_ADAPTER(DynamicGRUV2Grad)
DECLARE_OP_USE_OUTPUT(DynamicGRUV2Grad)

DECLARE_OP_ADAPTER(CommonLSTM)
DECLARE_OP_USE_OUTPUT(CommonLSTM)

DECLARE_OP_ADAPTER(GRUV2HiddenGradCell)
DECLARE_OP_USE_OUTPUT(GRUV2HiddenGradCell)

DECLARE_OP_ADAPTER(CommonGRU)
DECLARE_OP_USE_OUTPUT(CommonGRU)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_RNN_DECLARE_H_
