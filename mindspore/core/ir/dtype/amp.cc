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
#include "ir/dtype/amp.h"
#include "ir/dtype/number.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace amp {
std::map<std::string, PrimCastStrategyInfo> AmpStrategy::GetStrategyInfoCache() const { return strategy_info_cache_; }

void AmpStrategy::AddStrategyInfoToCache(const std::string &op_name, const PrimCastStrategyInfo &strategy_info) {
  strategy_info_cache_[op_name] = strategy_info;
}

bool AmpStrategy::operator==(const AmpStrategy &other) const {
  if (this == &other) {
    return true;
  }
  return enable_ == other.IsEnable() && amp_level_ == other.GetAmpLevel() && amp_dtype_ == other.GetAmpDtype() &&
         white_list_ == other.GetWhiteList() && black_list_ == other.GetBlackList();
}

std::string AmpStrategy::ToString() const {
  if (!enable_) {
    return "amp_strategy: empty";
  }
  static std::map<AmpLevel, std::string> amp_level_str_map = {
    {AmpLevel::O0, "O0"}, {AmpLevel::O1, "O1"}, {AmpLevel::O2, "O2"}, {AmpLevel::O3, "O3"}, {AmpLevel::Auto, "Auto"}};
  std::ostringstream oss;
  oss << "amp_strategy: " << amp_level_str_map[amp_level_];
  if (amp_dtype_ != nullptr) {
    oss << ", " << amp_dtype_->ToString();
  }
  return oss.str();
}
}  // namespace amp
}  // namespace mindspore
