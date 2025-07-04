/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_FRAMEWORK_OP_NAME_H_
#define MINDSPORE_CORE_BASE_FRAMEWORK_OP_NAME_H_
#include "ir/core_ops_name.h"
namespace mindspore {
constexpr auto kGatherOpName = "Gather";

// Attribute
constexpr auto kSetAttrOpName = "setattr";
constexpr auto kRintOpName = "Rint";
constexpr auto kReverseV2OpName = "ReverseV2";
constexpr auto kNoRepeatNGramOpName = "NoRepeatNGram";
constexpr auto kSearchSortedOpName = "SearchSorted";

// Meta Function Graph

// Others
constexpr auto kidentityOpName = "identity";
constexpr auto kEnvironSetOpName = "EnvironSet";
constexpr auto kEnvironGetOpName = "EnvironGet";
constexpr auto kEnvironAddOpName = "EnvironAdd";
constexpr auto kPopulationCountOpName = "PopulationCount";
constexpr auto kEnvironDestroyAllOpName = "EnvironDestroyAll";
constexpr auto kMutableOpName = "mutable";
constexpr auto kGetGradOpName = "GetGrad";
constexpr auto kSetSizeOpName = "SetSize";
constexpr auto kStringUpperOpName = "StringUpper";
constexpr auto kStringLowerOpName = "StringLower";
constexpr auto kHookBackwardName = "HookBackward";

// Framework
constexpr auto kSelectOpName = "Select";
constexpr auto kCallOpName = "call";
constexpr auto kMemCpyAsyncOpName = "memcpy_async";
constexpr auto kPrintOpName = "Print";
constexpr auto kPullOpName = "Pull";
constexpr auto kPyExecuteOpName = "PyExecute";
constexpr auto kPyInterpretOpName = "PyInterpret";
constexpr auto kPushOpName = "Push";
constexpr auto kQuantDTypeCastOpName = "QuantDTypeCast";
constexpr auto kRpcSendOpName = "RpcSend";
constexpr auto kTensorMoveOpName = "TensorMove";
constexpr auto kCheckValidOpName = "CheckValid";
constexpr auto kMakeDictOpName = "make_dict";
constexpr auto kSendOpName = "Send";
constexpr auto kReceiveOpName = "Receive";
constexpr auto kStreamSendOpName = "StreamSend";
constexpr auto kStreamRecvOpName = "StreamRecv";
constexpr auto kRaiseOpName = "raise";
constexpr auto kFormatOpName = "Format";
constexpr auto kMoveToOpName = "MoveTo";
constexpr auto kMoveAssignOpName = "MoveAssign";
constexpr auto kTraceGraphOpName = "TraceGraph";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_FRAMEWORK_OP_NAME_H_
