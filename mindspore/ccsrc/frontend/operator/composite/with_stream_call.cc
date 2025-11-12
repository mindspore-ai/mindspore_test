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

#include "frontend/operator/composite/with_stream_call.h"
#include "abstract/abstract_function.h"
#include "mindspore/core/include/ir/func_graph_flag.h"

namespace mindspore {
namespace prim {
FuncGraphPtr GetFuncFromAbstract(const abstract::AbstractBasePtr abs) {
  auto func_graph_abstract = dyn_cast<abstract::FuncGraphAbstractClosure>(abs);
  if (func_graph_abstract != nullptr) {
    auto func = func_graph_abstract->func_graph();
    return func;
  }
  return nullptr;
}

int64_t ExtractStreamId(const std::string &text) {
  std::string keyword = "stream id:";
  size_t pos = text.find(keyword);
  if (pos == std::string::npos) {
    return -1;
  }
  pos += keyword.length();
  while (pos < text.length() && std::isspace(text[pos])) {
    pos++;
  }
  int64_t result = 0;
  bool found_digit = false;

  while (pos < text.length() && std::isdigit(text[pos])) {
    found_digit = true;
    result = result * 10 + (text[pos] - '0');
    pos++;
  }
  return found_digit ? result : -1;
}

size_t GetStreamId(const ValuePtr &value) {
  auto stream_id = ExtractStreamId(value->ToString());
  if (stream_id == -1) {
    MS_LOG(EXCEPTION) << "GetStreamID node is wrong.";
  }
  return static_cast<size_t>(stream_id);
}

// WithStreamCall(mark_flag, func_graph, stream_id)
// eg:
// WithStreamCall("stream_id", body_func_graph, stream_id)
// WithStreamCall("stream_ctx_after", after_func_graph, stream_id)
// WithStreamCall("stream_limit_ctx_after", limit_after_func_graph, stream_id)
// or
// WithStreamCall(mark_flag, func_graph, stream_id, cube_num, vector_num)
// eg: WithStreamCall("stream_limit_id", body_func_graph, stream_id, cube_num, vector_num)
FuncGraphPtr WithStreamCall::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  size_t arg_length = args_abs_list.size();
  const size_t args_min_size = 3;
  const size_t args_max_size = 5;
  if (arg_length != args_min_size && arg_length != args_max_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "The WithStreamCall operator requires 3 or 5 arguments, but got " << arg_length
                               << ".";
  }
  constexpr auto kFlagIndex = 0;
  constexpr auto kFuncGraphIndex = 1;
  constexpr auto kStreamIdIndex = 2;
  auto flag_arg = args_abs_list[kFlagIndex];
  auto flag_str = GetValue<string>(flag_arg->BuildValue());
  auto func_abs = args_abs_list[kFuncGraphIndex];
  auto res_graph = GetFuncFromAbstract(func_abs);
  if (res_graph == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The first input of WithStreamCall operator must be func_graph, but got "
                               << func_abs->ToString();
  }

  ValuePtr value_track = args_abs_list[kStreamIdIndex]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(value_track);
  size_t stream_id = GetStreamId(value_track);
  res_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  if (arg_length == args_min_size) {
    res_graph->set_attr(flag_str, MakeValue(static_cast<size_t>(stream_id)));
  } else {
    constexpr auto kCubeNumIndex = 3;
    constexpr auto kVectorNumIndex = 4;
    auto cube_value = args_abs_list[kCubeNumIndex]->BuildValue();
    auto cube_num = GetValue<int64_t>(cube_value);
    auto vector_value = args_abs_list[kVectorNumIndex]->BuildValue();
    auto vector_num = GetValue<int64_t>(vector_value);
    res_graph->set_attr(kFuncGraphFlagStreamLimitId, MakeValue(static_cast<size_t>(stream_id)));
    res_graph->set_attr(kFuncGraphFlagCubeNum, MakeValue(static_cast<int64_t>(cube_num)));
    res_graph->set_attr(kFuncGraphFlagVectorNum, MakeValue(static_cast<int64_t>(vector_num)));
  }
  auto new_res_graph = std::make_shared<FuncGraph>();
  for (size_t index = 0; index < arg_length; ++index) {
    new_res_graph->add_parameter();
  }
  new_res_graph->set_output(NewValueNode(MakeValue("None")));
  return new_res_graph;
}
}  // namespace prim
}  // namespace mindspore
