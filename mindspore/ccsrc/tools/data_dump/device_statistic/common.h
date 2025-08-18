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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_COMMON_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_COMMON_H_

#include <chrono>
#include <string>
#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {

namespace datadump {

extern const char KStatMax[];
extern const char KStatMin[];
extern const char KStatMean[];
extern const char KStatL2Norm[];
extern const char KCheckOverflow[];

void WarningOnce(const std::string &device_name, const std::string &type_name, const std::string &statistic_name);

}  // namespace datadump
}  // namespace mindspore

#endif
