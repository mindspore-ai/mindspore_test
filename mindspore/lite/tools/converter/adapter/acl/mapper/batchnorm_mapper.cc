/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/batchnorm_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace lite {
BatchNormMapper::BatchNormMapper() : PrimitiveMapper(ops::kNameBatchNorm) {}
STATUS BatchNormMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  if (fmk_type == converter::kFmkTypeCaffe) {
    auto dst_prim = std::make_shared<acl::BNInference>();
    if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
      MS_LOG(ERROR) << "BatchNorm mapper failed.";
      return RET_ERROR;
    }
  }
  if (src_prim->HasAttr(ops::kFormat)) {
    // the attr format has been changed to data_format because of dynamic(defined in gen_lite_ops.h)
    src_prim->AddAttr(kAttrDataFormat, src_prim->GetAttr(ops::kFormat));
  }
  return RET_OK;
}

using mindspore::ops::kNameBatchNorm;
REGISTER_PRIMITIVE_MAPPER(kNameBatchNorm, BatchNormMapper)
}  // namespace lite
}  // namespace mindspore
