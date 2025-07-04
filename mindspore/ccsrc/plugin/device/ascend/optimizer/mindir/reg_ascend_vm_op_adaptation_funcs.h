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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_FUNCS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_FUNCS_H_

#include "include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore::opt {
bool ApplyAdagradV2PreCheck(const CNodePtr &node);
}  // namespace mindspore::opt

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_MINDIR_REG_ASCEND_VM_OP_ADAPTATION_FUNCS_H_
