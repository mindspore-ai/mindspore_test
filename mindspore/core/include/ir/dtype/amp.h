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
#ifndef MINDSPORE_CORE_IR_DTYPE_AMP_H_
#define MINDSPORE_CORE_IR_DTYPE_AMP_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "base/base.h"

namespace mindspore {
namespace amp {
// prim_name and arg indexes which need to be casted, e.g. (prim0, [arg0, arg1, ...])
// arg indexes empty means all float args need to be casted
using PrimArg = std::pair<std::string, std::vector<uint8_t>>;
using PrimArgList = std::vector<PrimArg>;

typedef enum AmpLevel {
  O0 = 0,
  O1 = 1,
  O2 = 2,
  O3 = 3,
  Auto = 4,
} AmpLevel;

typedef enum PrimCastStrategy {
  Ignore = 0,       // Do not insert cast for inputs
  DoCast = 1,       // Insert cast for inputs with specific float dtype in PrimCastInfo
  SetDtype = 2,     // Set prim dtype to specific float dtype in PrimCastInfo
  SetDtypeOpt = 3,  // Set prim dtype to specific float dtype in PrimCastInfo if dtype is not set by user
  AutoPromote = 4,  // Insert cast for inputs with widest float type
} PrimCastStrategy;

typedef struct PrimCastStrategyInfo {
  PrimCastStrategy strategy;     // cast strategy
  TypePtr dtype;                 // dtype that inputs to be casted to
  std::vector<uint8_t> arg_pos;  // position of args that need to be casted, cast all float args when empty
} PrimCastStrategyInfo;

class MS_CORE_API AmpStrategy {
 public:
  AmpStrategy() : enable_(false) {}

  AmpStrategy(const AmpLevel amp_level, const TypePtr amp_dtype, const PrimArgList white_list,
              const PrimArgList black_list)
      : amp_level_(amp_level), amp_dtype_(amp_dtype), white_list_(white_list), black_list_(black_list) {}
  ~AmpStrategy() = default;

  AmpLevel GetAmpLevel() const { return amp_level_; }
  TypePtr GetAmpDtype() const { return amp_dtype_; }
  PrimArgList GetWhiteList() const { return white_list_; }
  PrimArgList GetBlackList() const { return black_list_; }
  std::map<std::string, PrimCastStrategyInfo> GetStrategyInfoCache() const;
  void AddStrategyInfoToCache(const std::string &op_name, const PrimCastStrategyInfo &strategy_info);
  bool IsEnable() const { return enable_; }
  bool operator==(const AmpStrategy &other) const;
  std::string ToString() const;

 private:
  AmpLevel amp_level_ = AmpLevel::Auto;
  TypePtr amp_dtype_ = nullptr;
  PrimArgList white_list_;
  PrimArgList black_list_;
  bool enable_{true};
  std::map<std::string, PrimCastStrategyInfo> strategy_info_cache_;
};
using AmpStrategyPtr = std::shared_ptr<AmpStrategy>;
}  // namespace amp
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DTYPE_AMP_H_
