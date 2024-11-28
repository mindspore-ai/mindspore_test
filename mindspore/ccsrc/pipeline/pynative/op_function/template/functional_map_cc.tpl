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

#include "pipeline/pynative/op_function/auto_generate/functional_map.h"

#include <map>
#include <set>
#include <vector>
#include <string>
#include "frontend/operator/composite/functional_overload.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

namespace mindspore::ops {
${deprecated_method_decl}

std::map<std::string, std::vector<ValuePtr>> tensor_method_overload_map = {
  ${tensor_method_map}
};

std::map<std::string, std::vector<ValuePtr>> function_overload_map = {
  ${mint_map}
};

std::map<std::string, std::set<std::string>> tensor_method_kwonlyargs_map = {
  ${tensor_method_kwonlyargs_map}
};

std::map<std::string, std::set<std::string>> function_kwonlyargs_map = {
  ${mint_kwonlyargs_map}
};

std::map<std::string, std::vector<std::string>> tensor_method_overload_signature_map = {
  ${tensor_method_sigs_map}
};

std::map<std::string, std::vector<std::string>> function_overload_signature_map = {
  ${mint_sigs_map}
};
}  // namespace mindspore::ops