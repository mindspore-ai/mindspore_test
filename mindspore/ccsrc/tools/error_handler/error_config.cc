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
#include "tools/error_handler/error_config.h"
#include <mutex>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <iostream>
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace tools {
namespace {
// training fault tolerance config env var 'MS_ENABLE_TFT'
constexpr char kMsEnableTft[] = "MS_ENABLE_TFT";
constexpr char kTftKeyTre[] = "TRE";
constexpr char kTftKeyTreSnapShotSteps[] = "TRE_SNAPSHOT_STEPS";
constexpr char kTftValueNormalTRE[] = "1";
constexpr char kTftValueStepTRE[] = "2";
constexpr size_t kTftSubItemNumElems = 2;

// Trim from start (in place)
void LTrim(std::string *s, unsigned char extra = '\0') {
  s->erase(s->begin(), std::find_if(s->begin(), s->end(),
                                    [extra](unsigned char ch) { return !std::isspace(ch) && (ch != extra); }));
}

// Trim from end (in place)
void RTrim(std::string *s, unsigned char extra = '\0') {
  s->erase(
    std::find_if(s->rbegin(), s->rend(), [extra](unsigned char ch) { return !std::isspace(ch) && (ch != extra); })
      .base(),
    s->end());
}

// Trim from both ends (in place)
std::string &Trim(std::string *s) {
  LTrim(s);
  RTrim(s);
  return *s;
}

std::vector<std::string> SplitString(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::istringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}
}  // namespace

bool TftConfig::IsEnableTRE() {
  static bool enable_tre = []() {
    auto iter = GetConfigMap().find(kTftKeyTre);
    if (iter == GetConfigMap().end()) {
      MS_LOG(WARNING) << "Can find `" << kTftKeyTre << "` in environment var `" << kMsEnableTft << "`";
      return false;
    }
    if (iter->second == kTftValueNormalTRE) {
      return true;
    }
    return IsEnableStepTRE();
  }();
  return enable_tre;
}

bool TftConfig::IsEnableStepTRE() {
  static bool enable_step_tre = []() {
    auto iter = GetConfigMap().find(kTftKeyTre);
    if (iter == GetConfigMap().end()) {
      MS_LOG(WARNING) << "Can find `" << kTftKeyTre << "` in environment var `" << kMsEnableTft << "`";
      return false;
    }
    if (iter->second != kTftValueStepTRE) {
      MS_LOG(WARNING) << "Value of `" << kTftKeyTre << "` is `" << iter->second << "`, step tre is not enabled.";
      return false;
    }
    if (GetSnapShotSteps() <= 0) {
      MS_LOG(WARNING) << "Value of `" << kTftKeyTreSnapShotSteps << "` is `" << GetSnapShotSteps()
                      << "`, step tre is not enabled.";
      return false;
    }
    return true;
  }();
  return enable_step_tre;
}

int TftConfig::GetSnapShotSteps() {
  static int mem_cfg_steps = []() {
    auto iter = GetConfigMap().find(kTftKeyTreSnapShotSteps);
    if (iter == GetConfigMap().end()) {
      MS_LOG(WARNING) << "Can find `" << kTftKeyTreSnapShotSteps << "` in environment var `" << kMsEnableTft << "`";
      return 0;
    }
    try {
      return std::stoi(iter->second);
    } catch (std::invalid_argument const &ex) {
      MS_LOG(WARNING) << "Try to convert value " << iter->second << " for key " << kTftKeyTreSnapShotSteps
                      << " failed, what(): " << ex.what();
      return 0;
    } catch (std::out_of_range const &ex) {
      MS_LOG(WARNING) << "Try to convert value " << iter->second << " for key " << kTftKeyTreSnapShotSteps
                      << " failed, what(): " << ex.what();
      return 0;
    }
  }();
  return mem_cfg_steps;
}

std::map<std::string, std::string> &TftConfig::GetConfigMap() {
  static std::map<std::string, std::string> configs;
  static bool config_parsed = false;
  if (config_parsed) {
    return configs;
  }

  // parse value of MS_ENABLE_TFT in format '{key1:value1, key2:value2}'
  static std::once_flag flag;
  std::call_once(
    flag,
    [](std::map<std::string, std::string> &configs, bool &config_parsed) {
      auto tft_config = common::GetEnv(kMsEnableTft);
      MS_LOG(WARNING) << "Value of `" << kMsEnableTft << "` is `" << tft_config << "`";
      if (tft_config.empty()) {
        config_parsed = true;
        return;
      }
      // trim left '{' and right '}'
      LTrim(&tft_config, '{');
      RTrim(&tft_config, '}');
      MS_LOG(INFO) << "Trimmed value of `" << kMsEnableTft << "` is `" << tft_config << "`";

      std::vector<std::string> config_items = SplitString(tft_config, ',');
      for (const auto &cfg_item : config_items) {
        auto elems = SplitString(cfg_item, ':');
        if (elems.size() != kTftSubItemNumElems) {
          MS_LOG(WARNING) << "Ignore illegal item `" << cfg_item << "` in " << kMsEnableTft;
        }
        auto &key = Trim(&elems[0]);
        auto &value = Trim(&elems[1]);
        configs[key] = value;
        MS_LOG(WARNING) << "Insert key `" << key << "` with value `" << value << "`";
      }

      config_parsed = true;
    },
    configs, config_parsed);

  return configs;
}
}  // namespace tools
}  // namespace mindspore
