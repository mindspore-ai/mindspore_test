/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include "acl/acl_prof.h"

#include "utils/singleton.h"
#include "include/securec.h"
namespace mindspore {
namespace profiler {
namespace ascend {

enum class FeatureType {
  FEATURE_MIN = 0,
  FEATURE_ATTR,
  FEATURE_MEMORY_ACCESS,
  FEATURE_MAX,
};

struct FeatureInfo {
  char compatibility[16] = "\0";
  char featureVersion[16] = "\0";
  char affectedComponent[16] = "\0";
  char affectedComponentVersion[16] = "\0";
  char infoLog[128] = "\0";
  FeatureInfo() = default;
  FeatureInfo(const char *tempCompatibility, const char *tempFeatureVersion, const char *tempAffectedComponent,
              const char *tempAffectedComponentVersion, const char *tempInfoLog) {
    // 0 tempData, 1 structData
    std::vector<std::tuple<const char *, char *, size_t>> copyList = {
      {tempCompatibility, compatibility, sizeof(compatibility)},
      {tempFeatureVersion, featureVersion, sizeof(featureVersion)},
      {tempAffectedComponent, affectedComponent, sizeof(affectedComponent)},
      {tempAffectedComponentVersion, affectedComponentVersion, sizeof(affectedComponentVersion)},
      {tempInfoLog, infoLog, sizeof(infoLog)},
    };
    std::all_of(copyList.begin(), copyList.end(), [](std::tuple<const char *, char *, size_t> &copyNode) {
      const char *src = std::get<0>(copyNode);
      char *dest = std::get<1>(copyNode);
      size_t destSize = std::get<2>(copyNode);
      if (strcpy_s(dest, destSize, src) != 0) {
        return false;
      }
      return true;
    });
  }
  virtual ~FeatureInfo() {}
};

struct FeatureRecord {
  char featureName[64] = "\0";
  FeatureInfo info;
  FeatureRecord() = default;
  virtual ~FeatureRecord() {}
};

class FeatureMgr : public mindspore::Singleton<FeatureMgr> {
  friend class mindspore::Singleton<FeatureMgr>;

 public:
  FeatureMgr() = default;
  virtual ~FeatureMgr() {}
  void Init();
  bool IsSupportFeature(FeatureType featureName);

 private:
  void FormatFeatureList(size_t size, void *featuresData);
  bool IsTargetComponent(const char *component, const char *componentVersion);

 private:
  std::unordered_map<FeatureType, FeatureInfo> profFeatures_;
};

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
