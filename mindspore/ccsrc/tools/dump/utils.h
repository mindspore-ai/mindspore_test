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

#ifndef MINDSPORE_CCSRC_DEBUG_DUMP_UTILS_H_
#define MINDSPORE_CCSRC_DEBUG_DUMP_UTILS_H_

#include <string>

#include "utils/ms_utils.h"
#include "tools/visible.h"
#include "ir/tensor.h"

namespace mindspore {
namespace datadump {

DUMP_EXPORT std::uint32_t GetRankID();
DUMP_EXPORT bool StartsWith(const std::string &, const std::string &);
DUMP_EXPORT bool EndsWith(const std::string &, const std::string &);
DUMP_EXPORT void SaveTensor2NPY(std::string file_name, mindspore::tensor::TensorPtr tensor_ptr);

}  // namespace datadump
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_DUMP_UTILS_H_
