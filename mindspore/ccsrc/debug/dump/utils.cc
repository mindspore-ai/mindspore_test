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
#include "debug/dump/utils.h"
#include "utils/distributed_meta.h"

namespace mindspore {
namespace datadump {
std::uint32_t GetRankID() {
  std::uint32_t rank_id = 0;
  if (mindspore::DistributedMeta::GetInstance()->initialized()) {
    rank_id = mindspore::DistributedMeta::GetInstance()->global_rank_id();
  }
  return rank_id;
}

}  // namespace datadump
}  // namespace mindspore
