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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECT_HCCL_INIT_INFO_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECT_HCCL_INIT_INFO_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
namespace mindspore {
namespace distributed {
namespace collective {
class BACKEND_COMMON_EXPORT CollectHcclInitInfo {
 public:
  ~CollectHcclInitInfo() {}
  DISABLE_COPY_AND_ASSIGN(CollectHcclInitInfo);
  static std::shared_ptr<CollectHcclInitInfo> GetInstance();
  void SetBuffsize(const std::string &group_name, const uint32_t buffsize) {
    group_with_buffsize_[group_name] = buffsize;
    hccl_mem_size_ += (buffsize * 2);
  }
  void SetInitOrder(const std::string &group_name) { group_init_order_.emplace_back(group_name); }
  void SetRootInfo(const std::string &group_name, void *root_info) { group_with_root_info_[group_name] = root_info; }
  std::vector<std::string> GetInitOrder() { return group_init_order_; }
  uint32_t GetBuffsize(const std::string &group_name);
  void *GetRootInfo(const std::string &group_name);
  size_t GetHcclMemSize() { return hccl_mem_size_; }
  void Clear() {
    hccl_mem_size_ = 0;
    group_init_order_.clear();
    group_with_buffsize_.clear();
    group_with_root_info_.clear();
  }

 private:
  CollectHcclInitInfo() {
    hccl_mem_size_ = 0;
    group_init_order_.clear();
    group_with_buffsize_.clear();
    group_with_root_info_.clear();
  }
  std::vector<std::string> group_init_order_;
  std::unordered_map<std::string, uint32_t> group_with_buffsize_;
  std::unordered_map<std::string, void *> group_with_root_info_;
  size_t hccl_mem_size_;
};

}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECT_HCCL_INIT_INFO_H_
