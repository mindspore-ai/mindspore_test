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
#include <vector>
#include <string>
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

namespace mindspore::ops {
/*
${deprecated_method_decl}

std::map<std::string, std::vector<ValuePtr>> functional_method_map = {
  ${functional_method_map}
};
*/

std::map<std::string, std::vector<std::pair<std::string, std::string>>> functional_convert_map = {
  ${functional_map}
};

std::map<std::string, std::vector<std::string>> func_signature_map = {
  ${func_sigs_map}
};
}  // namespace mindspore::ops