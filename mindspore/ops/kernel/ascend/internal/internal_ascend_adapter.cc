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

#include "kernel/ascend/internal/internal_ascend_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
namespace mindspore {
namespace kernel {
aclError InternalAscendAdapter::AscendMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                                                  aclrtMemcpyKind kind, aclrtStream stream) {
  return CALL_ASCEND_API(aclrtMemcpyAsync, dst, destMax, src, count, ACL_MEMCPY_HOST_TO_DEVICE, stream);
}
}  // namespace kernel
}  // namespace mindspore
