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
#include "infer/list_inplace_clear.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/helper.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListInplaceClearInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  constexpr size_t input_len = 1;
  constexpr size_t data_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto data_abs = dyn_cast<abstract::AbstractList>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);

  abstract::AbstractListPtr ret;
  if (data_abs->dynamic_len()) {
    MS_LOG(INTERNAL_EXCEPTION) << "ListInplaceClear do not support dynamic length list input.";
  }
  abstract::AbstractBasePtrList empty_elements = {};
  ret = std::make_shared<abstract::AbstractList>(empty_elements);

  ret = AbstractBroaden(ret)->cast<abstract::AbstractListPtr>();
  ret->set_extra_info(data_abs->extra_info());

  return ret;
}
MIND_API_OPERATOR_IMPL(ListInplaceClear, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ListInplaceClear, prim::kPrimListInplaceClear, ListInplaceClearInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
