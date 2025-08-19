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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KENNEL_INTERNAL_INTERNAL_ASCEND_ADAPTER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KENNEL_INTERNAL_INTERNAL_ASCEND_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>

#include "include/backend/visible.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mindspore {
namespace kernel {
/**
 * InternalAscendAdapter is used to provide mindspore ascend api for plugin
 * because normal api symbol is deleted by mindspore os default
 */
class BACKEND_EXPORT InternalAscendAdapter {
 public:
  static aclError AscendMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind,
                                    aclrtStream stream);
};

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KENNEL_INTERNAL_INTERNAL_ASCEND_ADAPTER_H_
