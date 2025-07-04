/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_DEBUG_TRACE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_DEBUG_TRACE_H_

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <stack>
#include <deque>
#include <utility>

#include "utils/trace_base.h"
#include "utils/info.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "utils/any.h"

namespace mindspore {
namespace trace {
using TraceGraphEvalStack = std::deque<std::pair<abstract::AnalysisContextPtr, abstract::AnfNodeConfigPtr>>;
using TraceCNodeEvalStack = std::vector<abstract::AnfNodeConfigPtr>;
FRONTEND_EXPORT void TraceGraphEval();
FRONTEND_EXPORT void GetEvalStackInfo(std::ostringstream &oss);
void TraceGraphEvalEnter(const abstract::AnalysisContextPtr &context, const abstract::AnfNodeConfigPtr &node);
void TraceGraphEvalLeave(const abstract::AnalysisContextPtr &context);
void TraceGraphEvalStackPrepare(const TraceGraphEvalStack &graph_evals);
void TraceEvalCNodeStackPrepare(const TraceCNodeEvalStack &cnode_evals);
void TraceEvalCNodeEnter(const abstract::AnfNodeConfigPtr &node_config);
void TraceEvalCNodeLeave();
TraceCNodeEvalStack &GetCNodeDebugStack();
TraceGraphEvalStack &GetCurrentGraphEvalStack();
void GetTraceStackInfo(std::ostringstream &oss, bool add_title = false);
std::string GetAbstractStr(const abstract::AbstractBasePtr &abs);
FRONTEND_EXPORT void ClearTraceStack();
}  // namespace trace
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_TRACE_H_
