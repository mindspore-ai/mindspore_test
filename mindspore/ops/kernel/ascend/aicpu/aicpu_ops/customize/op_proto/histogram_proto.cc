/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "op_proto/inc/math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ----------------Histogram------------------
IMPLEMT_INFERFUNC(Histogram, HistogramInfer) {
  Shape shape;
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType out_dtype = x_dtype;
  int64_t bins;
  if (op.GetAttr("bins", bins) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr bins failed.");
    return GRAPH_FAILED;
  }
  if (Vector(bins, shape) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), string("fail to gen vector shape according bins."));
    return GRAPH_FAILED;
  }
  if (x_dtype != DT_INT32) {
    out_dtype = DT_FLOAT;
  }
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetDataType(out_dtype);
  output_desc.SetShape(shape);
  return op.UpdateOutputDesc("y", output_desc);
}

INFER_FUNC_REG(Histogram, HistogramInfer);
// ----------------Histogram End----------------------
}  // namespace ge
