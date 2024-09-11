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

#include "tools/converter/adapter/acl/mapper/scatter_nd_update_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "infer/tensor_copy.h"
#include "src/common/log_util.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/op_def/array_ops.h"

namespace mindspore {
namespace lite {
namespace {
const size_t kNumInputSize = 4;
const size_t kNumCnodeInputIndex1 = 1;  // var
const size_t kNumCnodeInputIndex3 = 3;  // updates
}  // namespace
STATUS ScatterNdUpdateMapper::Mapper(const CNodePtr &cnode) {
  if (cnode->size() != kNumInputSize) {
    MS_LOG(ERROR) << "cnode input size is " << cnode->size() << ", not equal kNumInputSize.";
    return RET_ERROR;
  }
  auto status = opt::AdjustInputToCnode(cnode, kNumCnodeInputIndex1);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "AdjustInputToCnode failed.";
    return RET_ERROR;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "GetValueNodeAndPrimFromCnode failed.";
    return RET_ERROR;
  }
  if (value_node == nullptr || src_prim == nullptr) {
    MS_LOG(ERROR) << "value_node or src_prim is nullptr.";
    return RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::ScatterNdUpdate>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make ScatterNdUpdate failed.";
    return RET_ERROR;
  }
  TypeId type_id;
  auto scale_input = cnode->inputs()[kNumCnodeInputIndex1];
  if (opt::GetDataTypeFromAnfNode(scale_input, &type_id) != RET_OK) {
    MS_LOG(ERROR) << "GetDataTypeFromAnfNode failed!";
    return RET_ERROR;
  }
  if (type_id == kNumberTypeBool && cnode->input(kNumCnodeInputIndex1)->abstract() != nullptr) {
    auto cast_fp16_node_1 = NewCNode(
      cnode, prim::kPrimCast, {cnode->input(kNumCnodeInputIndex1), NewValueNode(TypeIdToType(kNumberTypeFloat16))},
      cnode->input(kNumCnodeInputIndex1)->abstract()->Clone(), cnode->fullname_with_scope() + "_cast_fp16_1");
    if (cast_fp16_node_1 == nullptr) {
      MS_LOG(ERROR) << "Make CNode failed!";
      return RET_ERROR;
    }
    cnode->set_input(kNumCnodeInputIndex1, cast_fp16_node_1);

    auto cast_fp16_node_3 = NewCNode(
      cnode, prim::kPrimCast, {cnode->input(kNumCnodeInputIndex3), NewValueNode(TypeIdToType(kNumberTypeFloat16))},
      cnode->input(kNumCnodeInputIndex3)->abstract()->Clone(), cnode->fullname_with_scope() + "_cast_fp16_3");
    if (cast_fp16_node_3 == nullptr) {
      MS_LOG(ERROR) << "Make CNode failed!";
      return RET_ERROR;
    }
    cnode->set_input(kNumCnodeInputIndex3, cast_fp16_node_3);
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameScatterNdUpdate, ScatterNdUpdateMapper)
}  // namespace lite
}  // namespace mindspore
