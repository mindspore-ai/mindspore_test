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
#ifndef MINDSPORE_PI_JIT_UTILS_STOP_TRACE_REASON_H
#define MINDSPORE_PI_JIT_UTILS_STOP_TRACE_REASON_H

// stop trace reason enum
#define STOP_TRACE_REASON_ENUM                                                              \
  STOP_TRACE_REASON_KIND(NonStopTrace, "NonStopTrace")                                      \
  STOP_TRACE_REASON_KIND(StopTraceReasonUnknown, "Unknown Reason")                          \
  STOP_TRACE_REASON_KIND(StopTraceInfer_Fail, "Infer_Fail")                                 \
  STOP_TRACE_REASON_KIND(StopTraceLoop_Unsupported, "Loop_Unsupported")                     \
  STOP_TRACE_REASON_KIND(StopTraceIf_Unsupported, "If_Unsupported")                         \
  STOP_TRACE_REASON_KIND(StopTraceFunc_ArgType_Unsupported, "Func_ArgType_Unsupported")     \
  STOP_TRACE_REASON_KIND(StopTraceFunc_ArgHandle_Unsupported, "Func_ArgHandle_Unsupported") \
  STOP_TRACE_REASON_KIND(StopTraceFunc_Type_Unsupported, "Func_Type_Unsupported")           \
  STOP_TRACE_REASON_KIND(StopTraceRecurse_Unsupported, "Recurse_Unsupported")               \
  STOP_TRACE_REASON_KIND(StopTraceByteCode_Unsupported, "ByteCode_Unsupported")             \
  STOP_TRACE_REASON_KIND(StopTraceDataDependsOnGraphOut, "DataDependsOnGraphOut")           \
  STOP_TRACE_REASON_KIND(Trace_Fail, "Trace_Fail")                                          \
  STOP_TRACE_REASON_KIND(StopTraceSkip_Exception, "Skip_exception")                         \
  STOP_TRACE_REASON_KIND(StopTraceTensorHook, "StopTraceTensorHook")                        \
  STOP_TRACE_REASON_KIND(StopTraceUDReset, "UD_Reset")                                      \
  STOP_TRACE_REASON_KIND(StopTraceUDAnalyzeError, "UDAnalyzeError")                         \
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
