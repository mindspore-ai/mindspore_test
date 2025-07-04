/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_EXAMPLE_ASCEND_MEM_H
#define MINDSPORE_LITE_EXAMPLE_ASCEND_MEM_H
#include <string>
#include "acl/acl.h"

void *MallocDeviceMemory(size_t data_size) {
  void *device_data = nullptr;
  auto ret = aclrtMalloc(&device_data, data_size, ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ret != ACL_SUCCESS) {
    std::cerr << "Malloc device buffer failed , buffer size " << data_size;
    return nullptr;
  }
  return device_data;
}

void FreeDeviceMemory(void *device_data) {
  if (device_data) {
    aclrtFree(device_data);
  }
}

int CopyMemoryHost2Device(void *device_data, size_t dst_size, void *host_data, size_t src_size) {
  auto ret = aclrtMemcpy(device_data, dst_size, host_data, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    std::cerr << "Acl memcpy host data to device failed, src size: " << src_size << ", dst size: " << dst_size
              << std::endl;
    return -1;
  }
  return 0;
}

int CopyMemoryDevice2Host(void *host_data, size_t dst_size, void *device_data, size_t src_size) {
  auto ret = aclrtMemcpy(host_data, dst_size, device_data, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    std::cerr << "Acl memcpy device data to host failed, src size: " << src_size << ", dst size: " << dst_size
              << std::endl;
    return -1;
  }
  return 0;
}
#endif  // MINDSPORE_LITE_EXAMPLE_ASCEND_MEM_H
