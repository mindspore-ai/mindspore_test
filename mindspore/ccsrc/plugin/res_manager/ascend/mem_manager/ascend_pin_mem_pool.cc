/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/res_manager/ascend/mem_manager/ascend_pin_mem_pool.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendPinMemPool &AscendPinMemPool::GetInstance() {
  static AscendPinMemPool instance{};
  return instance;
}

void AscendPinMemPool::PinnedMemAlloc(DeviceMemPtr *addr, size_t alloc_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  AscendHalManager::GetInstance().SetContext(device_id);
  aclError rt_ret = CALL_ASCEND_API(aclrtMallocHost, addr, alloc_size);
  if ((rt_ret != ACL_SUCCESS) || (*addr == nullptr)) {
    MS_LOG(ERROR) << "PinMemPool aclrtMallocHost failed.";
    return;
  }
  MS_LOG(INFO) << "Enable pinned memory success addr:" << *addr << " size:" << alloc_size;
}

bool AscendPinMemPool::FreeDeviceMem(const DeviceMemPtr &addr) {
  MS_EXCEPTION_IF_NULL(addr);
  if (pinned_mem_) {
    (void)CALL_ASCEND_API(aclrtFreeHost, addr);
  } else {
    free(addr);
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
