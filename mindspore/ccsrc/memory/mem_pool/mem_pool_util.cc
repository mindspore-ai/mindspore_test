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
#include "include/backend/mem_reuse/mem_pool_util.h"

#include "include/common/debug/common.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace memory {
namespace mem_pool {
const std::map<MemType, std::string> kMemTypeStr = {{MemType::kWeight, "Weight"},
                                                    {MemType::kConstantValue, "ConstantValue"},
                                                    {MemType::kKernel, "Kernel"},
                                                    {MemType::kGraphOutput, "GraphOutput"},
                                                    {MemType::kSomas, "Somas"},
                                                    {MemType::kSomasOutput, "SomasOutput"},
                                                    {MemType::kGeConst, "GeConst"},
                                                    {MemType::kGeFixed, "GeFixed"},
                                                    {MemType::kBatchMemory, "BatchMemory"},
                                                    {MemType::kContinuousMemory, "ContinuousMemory"},
                                                    {MemType::kPyNativeInput, "PyNativeInput"},
                                                    {MemType::kPyNativeOutput, "PyNativeOutput"},
                                                    {MemType::kWorkSpace, "WorkSpace"},
                                                    {MemType::kOther, "Other"}};

std::string MemTypeToStr(MemType mem_type) { return kMemTypeStr.at(mem_type); }

std::string GeneratePath(size_t rank_id, const std::string &file_name, const std::string &suffix) {
  std::string path;
  auto &&ms_context = MsContext::GetInstance();
  path = ms_context->get_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH);
  if (path.empty()) {
    path = "./";
  }
  if (path.back() != '/') {
    path += "/";
  }

  if (rank_id != SIZE_MAX) {
    path += "rank_" + std::to_string(rank_id) + "/";

    auto path_opt = Common::CreatePrefixPath(path);
    if (!path_opt.has_value()) {
      MS_LOG(ERROR) << "Create path : " << path << " failed.";
    }
  }

  if (!file_name.empty()) {
    path += file_name;
  }
  if (!suffix.empty()) {
    path += "." + suffix;
  }

  auto file_path_opt = Common::CreatePrefixPath(path);
  if (!file_path_opt.has_value()) {
    MS_LOG(WARNING) << "Generate path for rank id : " << rank_id << ", file_name : " << file_name
                    << ", suffix : " << suffix << "failed.";
    return "";
  }
  ChangeFileMode(path, S_IWUSR | S_IRUSR);
  return path;
}
}  // namespace mem_pool
}  // namespace memory
}  // namespace mindspore
