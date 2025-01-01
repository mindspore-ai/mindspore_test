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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_

#include <map>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "ops/op_def.h"
#include "include/common/visible.h"

namespace mindspore {
namespace prim {
class DeprecatedTensorMethod : public MetaFuncGraph {
 public:
  explicit DeprecatedTensorMethod(const std::string &name, const std::string &method)
      : MetaFuncGraph(name), method_(method) {}
  ~DeprecatedTensorMethod() override = default;
  MS_DECLARE_PARENT(DeprecatedTensorMethod, MetaFuncGraph)
  std::string method() const { return method_; }
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &) override;

 private:
  std::string method_;
};
using DeprecatedTensorMethodPtr = std::shared_ptr<DeprecatedTensorMethod>;

bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name);
std::map<size_t, std::pair<ValuePtr, bool>> &GetFunctionalConvertCache();
std::string BuildArgsTypeString(const TypePtr &arg_abs);
FRONTEND_EXPORT std::string BuildFunctionalErrorMsg(const std::string &function_name,
                                                    const std::vector<std::string> &arg_info_list, bool is_method);
FRONTEND_EXPORT std::stringstream BuildApiInputInfo(const std::string &function_name,
                                                    const std::vector<std::string> &arg_info_list);
AnfNodePtr ConvertFunctionalToPrimitive(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                        bool is_method);
AnfNodePtr ConvertFunctionalToPyExecute(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        bool is_method);
}  // namespace prim
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_FUNCTIONAL_OVERLOAD_H_
