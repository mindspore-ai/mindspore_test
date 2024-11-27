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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_

#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
TypeId GetSingleNodeOutputTypeId(const mindspore::AnfNodePtr &node);
AnfNodePtr GetCastNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node, const TypeId &dst_type_id);
AnfNodePtr GetReshapeNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                          const ShapeVector &dst_shape);
AnfNodePtr GetBroadcastToNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &node,
                              const ShapeVector &dst_shape);
AnfNodePtr GetMatMulNode(const FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input,
                         const mindspore::AnfNodePtr &other, const bool &transpose_a, const bool &transpose_b);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_IMPORT_CONVERT_EXTEND_OPS_UTILS_H_
