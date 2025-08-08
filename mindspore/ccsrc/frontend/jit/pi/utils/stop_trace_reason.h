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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_STOP_TRACE_REASON_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_STOP_TRACE_REASON_H

// stop trace reason enum
#define STOP_TRACE_REASON_ENUM                                                                                      \
  STOP_TRACE_REASON_KIND(NonStopTrace, "NonStopTrace")                                                              \
  STOP_TRACE_REASON_KIND(StopTraceReasonUnknown, "Cannot infer the execution result")                               \
  STOP_TRACE_REASON_KIND(StopTraceLoop_Unsupported, "Loop control flow is not supported")                           \
  STOP_TRACE_REASON_KIND(StopTraceLoop_Failed, "Trace loop control flow failed")                                    \
  STOP_TRACE_REASON_KIND(StopTraceLoop_IterableType_Unsupported, "Unsupported iterable type for loop control flow") \
  STOP_TRACE_REASON_KIND(StopTraceIf_Unsupported, "Data-dependent conditional control flow is not supported")       \
  STOP_TRACE_REASON_KIND(StopTraceFunc_ArgHandle_Unsupported, "Unsupported function argument")                      \
  STOP_TRACE_REASON_KIND(StopTraceFunc_Type_Unsupported, "Unsupported function")                                    \
  STOP_TRACE_REASON_KIND(StopTraceFunc_Trace_Fail, "Trace function call failed")                                    \
  STOP_TRACE_REASON_KIND(StopTraceByteCode_Unsupported, "Unsupported bytecode")                                     \
  STOP_TRACE_REASON_KIND(StopTraceFreeVar_Modify_Unsupported,                                                       \
                         "Assignment or deletion of free variables is not supported")                               \
  STOP_TRACE_REASON_KIND(StopTraceNoGraphCaptured, "No graph captured")                                             \
  STOP_TRACE_REASON_KIND(StopTraceSkip_Exception,                                                                   \
                         "with blocks, try-except-finally blocks, and exception raising are not supported")         \
  STOP_TRACE_REASON_KIND(StopTraceGraphOutput_Type_Unsupported,                                                     \
                         "This data type is not supported as an output of graph")                                   \
  STOP_TRACE_REASON_KIND(StopTraceUDAnalyze_Error, "Framework error in the UD analysis")                            \
  STOP_TRACE_REASON_KIND(StopTraceConstantFold_Failed, "Constant-fold failed")                                      \
  STOP_TRACE_REASON_KIND(StopTraceNamedtuple_Getattr_Failed, "Namedtuple getattr failed")                           \
  STOP_TRACE_REASON_KIND(StopTraceReadDeletedGlobalVariable, "Attempt to read a deleted global variable")           \
  STOP_TRACE_REASON_KIND(StopTraceReadDeletedAttr, "Attempt to read a deleted attribute")                           \
  STOP_TRACE_REASON_KIND(StopTraceYieldFromIterator_Unsupported, "Yield from iterator is not supported")            \
  STOP_TRACE_REASON_KIND(StopTraceDataType_Unsupported, "Unsupported data type in graph")                           \
  STOP_TRACE_REASON_KIND(StopTraceCanNotCreateCell, "Can not create nn.Cell in graph")                              \
  STOP_TRACE_REASON_KIND(StopTrace_Reason_Count, "StopTrace_Reason_Count")

enum StopTraceReason {
#define STOP_TRACE_REASON_KIND(kind, desc) k##kind,
  STOP_TRACE_REASON_ENUM
#undef STOP_TRACE_REASON_KIND
};

constexpr const char *GetStopTraceReasonDesc(StopTraceReason res) {
#define STOP_TRACE_REASON_KIND(kind, desc) \
  if (StopTraceReason::k##kind == res) {   \
    return desc;                           \
  }

  STOP_TRACE_REASON_ENUM
#undef STOP_TRACE_REASON_KIND
  return "???";
}
#undef STOP_TRACE_REASON_ENUM

#endif
