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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_INLINE_REASON_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_UTILS_INLINE_REASON_H

// inline reason enum
#define INLINE_REASON_ENUM                                                           \
  INLINE_REASON_KIND(InlineUnknown, "Unknown")                                       \
  INLINE_REASON_KIND(Inline, "Inline")                                               \
  INLINE_REASON_KIND(InlinePartial, "InlinePartial")                                 \
  INLINE_REASON_KIND(InlineDisabledInCfg, "DisabledInCfg")                           \
  INLINE_REASON_KIND(InlineTooDeep, "TooDeep")                                       \
  INLINE_REASON_KIND(InlineInfer_Fail, "Infer_Fail")                                 \
  INLINE_REASON_KIND(InlineCFunction_Unsupported, "CFunction_Unsupported")           \
  INLINE_REASON_KIND(InlineGraphSupportedByMS, "GraphSupportedByMS")                 \
  INLINE_REASON_KIND(InlineFunc_ArgType_Unsupported, "Func_ArgType_Unsupported")     \
  INLINE_REASON_KIND(InlineFunc_ArgHandle_Unsupported, "Func_ArgHandle_Unsupported") \
  INLINE_REASON_KIND(InlineFunc_ArgType_IsClass, "Func_ArgType_IsClass")             \
  INLINE_REASON_KIND(InlineFunc_Type_Unsupported, "Func_Type_Unsupported")           \
  INLINE_REASON_KIND(InlineFuncSpecialize, "FuncSpecialize")                         \
  INLINE_REASON_KIND(InlineRecurse_Unsupported, "Recurse_Unsupported")               \
  INLINE_REASON_KIND(InlineFunc_NoneTensor, "InlineFunc_NoneTensor")                 \
  INLINE_REASON_KIND(InlinePolicyDisabled, "InlinePolicyDisabled")                   \
  INLINE_REASON_KIND(Inline_Reason_Count, "Inline_Reason_Count")

enum InlineReason {
#define INLINE_REASON_KIND(kind, desc) k##kind,
  INLINE_REASON_ENUM
#undef INLINE_REASON_KIND
};

constexpr const char *GetInlineReasonDesc(InlineReason res) {
#define INLINE_REASON_KIND(kind, desc) \
  if (InlineReason::k##kind == res) {  \
    return desc;                       \
  }

  INLINE_REASON_ENUM
#undef INLINE_REASON_KIND
  return "???";
}
#undef INLINE_REASON_ENUM

#endif
