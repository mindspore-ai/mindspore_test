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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_PS_STATIC_ANALYSIS_FUNCTIONAL_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_PS_STATIC_ANALYSIS_FUNCTIONAL_UTILS_H_

#include <map>
#include <utility>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/dtype.h"
#include "ops/op_def.h"

namespace mindspore {
/* namespace to support prim related definition */
namespace prim {
bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name);
ValuePtr GetTensorPyMethod(const std::string &prim_name, const std::string &method_name);
std::map<size_t, std::pair<std::string, std::string>> &GetFunctionalConvertCache();
std::pair<std::string, std::string> ConvertFunctionalToPrimitive(const std::string &functional_name,
                                                                 const abstract::AbstractBasePtrList &args_abs_list);
AnfNodePtr ConvertFunctionalToPyExecute(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode);
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_PS_STATIC_ANALYSIS_FUNCTIONAL_UTILS_H_
