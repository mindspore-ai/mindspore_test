/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_INCLUDE_IR_CORE_OPS_NAME_H
#define MINDSPORE_CORE_INCLUDE_IR_CORE_OPS_NAME_H
namespace mindspore {
constexpr auto kDependOpName = "Depend";
constexpr auto kJOpName = "J";
constexpr auto kVmapOpName = "Vmap";
constexpr auto kTaylorOpName = "Taylor";
constexpr auto kEnvironCreateOpName = "EnvironCreate";
constexpr auto kLoadOpName = "Load";
constexpr auto kIsInstanceOpName = "isinstance";
constexpr auto kUpdateStateOpName = "UpdateState";
constexpr auto kVirtualViewGradOpName = "_VirtualViewGrad";
constexpr auto kReturnOpName = "Return";
constexpr auto kSwitchOpName = "Switch";
constexpr auto kMakeTupleOpName = "MakeTuple";
constexpr auto kTupleGetItemOpName = "TupleGetItem";
constexpr auto kListGetItemOpName = "list_getitem";
constexpr auto kMakeListNewOpName = "make_list";
constexpr auto kPrimConditionSwitchOpName = "ConditionSwitch";
constexpr auto kPrimConditionGatherOpName = "ConditionGather";
}  // namespace mindspore
#endif  // MINDSPORE_CORE_INCLUDE_IR_CORE_OPS_NAME_H
