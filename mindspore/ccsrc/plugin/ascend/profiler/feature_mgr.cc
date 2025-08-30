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
#include "plugin/ascend/profiler/feature_mgr.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_prof_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace profiler {
namespace ascend {

namespace {
static const char *VERSION = "2.5.0\0";
static const int ACL_ERROR_PROF_MODULES_UNSUPPORTED = 200007;
static const int ACL_SUCCESS = 0;

static std::unordered_map<std::string, FeatureType> NAME_TABLE = {{"ATTR", FeatureType::FEATURE_ATTR},
                                                                  {"MemoryAccess", FeatureType::FEATURE_MEMORY_ACCESS}};

// featureName, featureVersion
static std::unordered_map<FeatureType, std::string> FMK_FEATURES = {{FeatureType::FEATURE_ATTR, "1"},
                                                                    {FeatureType::FEATURE_MEMORY_ACCESS, "1"}};
}  // namespace

void FeatureMgr::Init() {
  size_t size = 0;
  void *dataPtr = nullptr;
  if (mindspore::device::ascend::aclprofGetSupportedFeatures_ == nullptr &&
      mindspore::device::ascend::aclprofGetSupportedFeaturesV2_ == nullptr) {
    MS_LOG(WARNING) << "CANN not support to get aclprofGetSupportedFeatures or aclprofGetSupportedFeaturesV2_ method.";
    return;
  }
  auto ret = (mindspore::device::ascend::aclprofGetSupportedFeaturesV2_ != nullptr)
               ? CALL_ASCEND_API(aclprofGetSupportedFeaturesV2, &size, &dataPtr)
               : CALL_ASCEND_API(aclprofGetSupportedFeatures, &size, &dataPtr);
  if (ret == ACL_ERROR_PROF_MODULES_UNSUPPORTED) {
    MS_LOG(EXCEPTION) << "Not support to get feature list.";
    return;
  } else if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed to get feature list.";
    return;
  }

  FormatFeatureList(size, dataPtr);
}

void FeatureMgr::FormatFeatureList(size_t size, void *featuresData) {
  FeatureRecord *features = static_cast<FeatureRecord *>(featuresData);
  size_t i = 0;
  while ((features != nullptr) && (i < size)) {
    if (!IsTargetComponent(features->info.affectedComponent, features->info.affectedComponentVersion)) {
      MS_LOG(WARNING) << "feature: " << features->featureName << ", component is: " << features->info.affectedComponent
                      << ", componentVersion is: " << features->info.affectedComponentVersion;
      features++;
      i++;
      continue;
    }
    std::string featureName = features->featureName;
    auto it = NAME_TABLE.find(featureName);
    if (it == NAME_TABLE.end()) {
      MS_LOG(WARNING) << "Do not support feature: " << features->featureName << ", log is: " << features->info.infoLog;
      features++;
      i++;
      continue;
    }
    auto tempInfo =
      FeatureInfo(features->info.compatibility, features->info.featureVersion, features->info.affectedComponent,
                  features->info.affectedComponentVersion, features->info.infoLog);
    if (tempInfo.compatibility[0] == '\0' || tempInfo.featureVersion[0] == '\0' ||
        tempInfo.affectedComponent[0] == '\0' || tempInfo.affectedComponentVersion[0] == '\0' ||
        tempInfo.infoLog[0] == '\0') {
      MS_LOG(EXCEPTION) << "Create feature info failed, feature name is: " << features->featureName;
      features++;
      i++;
      continue;
    }
    profFeatures_[NAME_TABLE[featureName]] = tempInfo;
    features++;
    i++;
  }
}

bool FeatureMgr::IsTargetComponent(const char *component, const char *componentVersion) {
  if (strcmp(component, "all") == 0) {
    return true;
  }
  if (strcmp(component, "mindspore") == 0 && strcmp(componentVersion, VERSION) == 0) {
    return true;
  }
  return false;
}

bool FeatureMgr::IsSupportFeature(FeatureType featureName) {
  auto fmkIt = FMK_FEATURES.find(featureName);
  auto profIt = profFeatures_.find(featureName);
  if (fmkIt == FMK_FEATURES.end() || profIt == profFeatures_.end()) {
    MS_LOG(WARNING) << "FMW or CANN do not support this feature type is: " << featureName;
    return false;
  }

  std::string featureVersion = profFeatures_[featureName].featureVersion;
  if (FMK_FEATURES[featureName] > featureVersion) {
    return false;
  } else if (FMK_FEATURES[featureName] < featureVersion) {
    return (strcmp(profFeatures_[featureName].compatibility, "1") == 0);
  }
  return true;
}

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
