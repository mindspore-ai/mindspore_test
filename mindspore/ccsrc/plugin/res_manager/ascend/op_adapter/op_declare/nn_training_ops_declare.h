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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_

#include "plugin/res_manager/ascend/op_adapter/custom_op_proto/cust_nn_training.h"
#include "plugin/res_manager/ascend/op_adapter/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(ApplyAdam)
DECLARE_OP_USE_OUTPUT(ApplyAdam)

DECLARE_OP_ADAPTER(ApplyAdamWMS)
DECLARE_OP_USE_OUTPUT(ApplyAdamWMS)

DECLARE_OP_ADAPTER(ApplyAdamD)
DECLARE_OP_USE_OUTPUT(ApplyAdamD)

DECLARE_OP_ADAPTER(ApplyAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyAdagradD)

DECLARE_OP_ADAPTER(ApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(ApplyAdagradV2D)

DECLARE_OP_ADAPTER(ApplyAddSignD)
DECLARE_OP_USE_OUTPUT(ApplyAddSignD)

DECLARE_OP_ADAPTER(SparseApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradV2D)

DECLARE_OP_ADAPTER(DataFormatDimMap)
DECLARE_OP_USE_OUTPUT(DataFormatDimMap)

DECLARE_OP_ADAPTER(ApplyAdadeltaD)
DECLARE_OP_USE_OUTPUT(ApplyAdadeltaD)

DECLARE_OP_ADAPTER(ApplyAdaMaxD)
DECLARE_OP_USE_OUTPUT(ApplyAdaMaxD)

DECLARE_OP_ADAPTER(ApplyGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyGradientDescent)

DECLARE_OP_ADAPTER(ApplyPowerSignD)
DECLARE_OP_USE_OUTPUT(ApplyPowerSignD)

DECLARE_OP_ADAPTER(ApplyProximalGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyProximalGradientDescent)

DECLARE_OP_ADAPTER(SGD)
DECLARE_OP_USE_OUTPUT(SGD)

DECLARE_OP_ADAPTER(ApplyMomentum)
DECLARE_OP_USE_OUTPUT(ApplyMomentum)

DECLARE_OP_ADAPTER(SparseApplyAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradD)

DECLARE_OP_ADAPTER(ApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyProximalAdagradD)

DECLARE_OP_ADAPTER(SparseApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyProximalAdagradD)

DECLARE_OP_ADAPTER(LarsV2Update)
DECLARE_OP_USE_OUTPUT(LarsV2Update)

DECLARE_OP_ADAPTER(ApplyFtrl)
DECLARE_OP_USE_OUTPUT(ApplyFtrl)

DECLARE_OP_ADAPTER(SparseApplyFtrlD)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlD)

DECLARE_OP_ADAPTER(SparseApplyFtrl)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrl)

DECLARE_OP_ADAPTER(SparseApplyFtrlV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlV2D)

DECLARE_OP_ADAPTER(ApplyRMSPropD)
DECLARE_OP_USE_INPUT_ATTR(ApplyRMSPropD)
DECLARE_OP_USE_OUTPUT(ApplyRMSPropD)

DECLARE_OP_ADAPTER(ApplyCenteredRMSProp)
DECLARE_OP_USE_OUTPUT(ApplyCenteredRMSProp)

DECLARE_OP_ADAPTER(SparseApplyRMSProp)
DECLARE_OP_USE_OUTPUT(SparseApplyRMSProp)

DECLARE_OP_ADAPTER(SparseApplyRMSPropD)
DECLARE_OP_USE_OUTPUT(SparseApplyRMSPropD)

DECLARE_OP_ADAPTER(SparseApplyAdagrad)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagrad)

DECLARE_OP_ADAPTER(ApplyKerasMomentumD)
DECLARE_OP_USE_OUTPUT(ApplyKerasMomentumD)

DECLARE_OP_ADAPTER(ApplyAdamWithAmsgradV2)
DECLARE_OP_USE_OUTPUT(ApplyAdamWithAmsgradV2)

DECLARE_OP_ADAPTER(ApplyAdamWithAmsgradD)
DECLARE_OP_USE_OUTPUT(ApplyAdamWithAmsgradD)

DECLARE_OP_ADAPTER(ApplyAdagrad)
DECLARE_OP_USE_OUTPUT(ApplyAdagrad)

DECLARE_OP_ADAPTER(ApplyAdagradDA)
DECLARE_OP_USE_OUTPUT(ApplyAdagradDA)

DECLARE_OP_ADAPTER(ApplyRMSProp)
DECLARE_OP_USE_OUTPUT(ApplyRMSProp)

DECLARE_OP_ADAPTER(ApplyProximalAdagrad)
DECLARE_OP_USE_OUTPUT(ApplyProximalAdagrad)

DECLARE_OP_ADAPTER(SparseApplyProximalAdagrad)
DECLARE_OP_USE_OUTPUT(SparseApplyProximalAdagrad)

DECLARE_OP_ADAPTER(ApplyAdadelta)
DECLARE_OP_USE_OUTPUT(ApplyAdadelta)

DECLARE_OP_ADAPTER(SparseApplyAdadelta)
DECLARE_OP_USE_OUTPUT(SparseApplyAdadelta)

DECLARE_CUST_OP_ADAPTER(FusedSparseProximalAdagrad)
DECLARE_CUST_OP_USE_OUTPUT(FusedSparseProximalAdagrad)

DECLARE_CUST_OP_ADAPTER(FusedSparseFtrl)
DECLARE_CUST_OP_USE_OUTPUT(FusedSparseFtrl)

DECLARE_CUST_OP_ADAPTER(FusedSparseAdam)
DECLARE_CUST_OP_USE_OUTPUT(FusedSparseAdam)

DECLARE_CUST_OP_ADAPTER(FusedSparseLazyAdam)
DECLARE_CUST_OP_USE_OUTPUT(FusedSparseLazyAdam)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_
