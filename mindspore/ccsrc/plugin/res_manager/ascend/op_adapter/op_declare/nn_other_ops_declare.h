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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_

#include "op_proto/inc/nn_other.h"
#include "plugin/res_manager/ascend/op_adapter/op_declare/op_declare_macro.h"

// ApplyRotaryPosEmb
DECLARE_OP_ADAPTER(ApplyRotaryPosEmb)
DECLARE_OP_USE_OUTPUT(ApplyRotaryPosEmb)

// InitPartitionMap
DECLARE_OP_ADAPTER(InitPartitionMap)
DECLARE_OP_USE_OUTPUT(InitPartitionMap)

// InitEmbeddingHashmap
DECLARE_OP_ADAPTER(InitEmbeddingHashmap)
DECLARE_OP_USE_OUTPUT(InitEmbeddingHashmap)

// EmbeddingTableImport
DECLARE_OP_ADAPTER(EmbeddingTableImport)
DECLARE_OP_USE_OUTPUT(EmbeddingTableImport)

// EmbeddingTableFind
DECLARE_OP_ADAPTER(EmbeddingTableFind)
DECLARE_OP_USE_OUTPUT(EmbeddingTableFind)

// EmbeddingTableFindAndInit
DECLARE_OP_ADAPTER(EmbeddingTableFindAndInit)
DECLARE_OP_USE_OUTPUT(EmbeddingTableFindAndInit)

// EmbeddingApplySgd
DECLARE_OP_ADAPTER(EmbeddingApplySgd)
DECLARE_OP_USE_OUTPUT(EmbeddingApplySgd)

// EmbeddingApplyRmsprop
DECLARE_OP_ADAPTER(EmbeddingApplyRmsprop)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyRmsprop)

// EmbeddingApplyFtrl
DECLARE_OP_ADAPTER(EmbeddingApplyFtrl)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyFtrl)

// EmbeddingApplyAdam
DECLARE_OP_ADAPTER(EmbeddingApplyAdam)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdam)

// EmbeddingApplyAdamW
DECLARE_OP_ADAPTER(EmbeddingApplyAdamW)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdamW)

// EmbeddingApplyAdaGrad
DECLARE_OP_ADAPTER(EmbeddingApplyAdaGrad)
DECLARE_OP_USE_OUTPUT(EmbeddingApplyAdaGrad)

// EmbeddingComputeVarImport
DECLARE_OP_ADAPTER(EmbeddingComputeVarImport)
DECLARE_OP_USE_OUTPUT(EmbeddingComputeVarImport)

// EmbeddingComputeVarExport
DECLARE_OP_ADAPTER(EmbeddingComputeVarExport)
DECLARE_OP_USE_OUTPUT(EmbeddingComputeVarExport)

// EmbeddingTableExport
DECLARE_OP_ADAPTER(EmbeddingTableExport)
DECLARE_OP_USE_OUTPUT(EmbeddingTableExport)

// FakeRemoteLookupUniqued
DECLARE_OP_ADAPTER(FakeRemoteLookupUniqued)
DECLARE_OP_USE_OUTPUT(FakeRemoteLookupUniqued)

// EmbeddingTableEvict
DECLARE_OP_ADAPTER(EmbeddingTableEvict)
DECLARE_OP_USE_OUTPUT(EmbeddingTableEvict)

// EmbeddingFeatureMappingV2
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingV2)
DECLARE_OP_USE_OUTPUT(EmbeddingFeatureMappingV2)

// EmbeddingFeatureMappingTableSize
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingTableSize)
DECLARE_OP_USE_OUTPUT(EmbeddingFeatureMappingTableSize)

// EmbeddingFeatureMappingFind
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingFind)
DECLARE_OP_USE_DYN_OUTPUT(EmbeddingFeatureMappingFind)

// EmbeddingFeatureMappingExport
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingExport)
DECLARE_OP_USE_DYN_INPUT(EmbeddingFeatureMappingExport)

// EmbeddingFeatureMappingFileSize
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingFileSize)
DECLARE_OP_USE_OUTPUT(EmbeddingFeatureMappingFileSize)

// EmbeddingFeatureMappingImport
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingImport)
DECLARE_OP_USE_DYN_OUTPUT(EmbeddingFeatureMappingImport)

// EmbeddingFeatureMappingInsert
DECLARE_OP_ADAPTER(EmbeddingFeatureMappingInsert)
DECLARE_OP_USE_DYN_INPUT(EmbeddingFeatureMappingInsert)

// RotaryPositionEmbedding
DECLARE_OP_ADAPTER(RotaryPositionEmbedding)
DECLARE_OP_USE_OUTPUT(RotaryPositionEmbedding)

// RotaryPositionEmbeddingGrad
DECLARE_OP_ADAPTER(RotaryPositionEmbeddingGrad)
DECLARE_OP_USE_OUTPUT(RotaryPositionEmbeddingGrad)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_OTHER_OPS_DECLARE_H_
