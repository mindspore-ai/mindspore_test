/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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
#include <algorithm>
#include <unordered_map>
#include "custom_op_proto/cust_math_ops.h"
#include "op_proto/inc/math_ops.h"
#include "op_proto/inc/ragged_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/reduce_infer_util.h"

namespace ge {

ONE_IN_ONE_OUT_INFER(Conj, input, output);
ONE_IN_ONE_OUT_INFER(Digamma, x, y);
CUST_ONE_IN_ONE_OUT_INFER(Polygamma, x, y);
TWO_IN_ONE_OUT_INFER(Igamma, a, x, z);
TWO_IN_ONE_OUT_INFER(Igammac, a, x, z);

// ----------------Angle-------------------
IMPLEMT_INFERFUNC(Angle, AngleInfer) {
  TensorDesc x_desc = op.GetInputDescByName("input");
  DataType x_type = x_desc.GetDataType();
  DataType out_type;
  switch (x_type) {
    case DT_COMPLEX64:
      out_type = DT_FLOAT;
      break;
    case DT_COMPLEX128:
      out_type = DT_DOUBLE;
      break;
    default:
      OP_LOGE(TbeGetName(op).c_str(), "Invalid input dtype: %s", DTypeStr(x_type).c_str());
      return GRAPH_FAILED;
  }
  x_desc.SetDataType(out_type);
  return op.UpdateOutputDesc("output", x_desc);
}

INFER_FUNC_REG(Angle, AngleInfer);
// ----------------Angle End-------------------

// ----------------ComplexAbs-------------------
IMPLEMT_INFERFUNC(ComplexAbs, ComplexAbsInfer) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  DataType x_type = x_desc.GetDataType();
  DataType out_type;
  switch (x_type) {
    case DT_COMPLEX64:
      out_type = DT_FLOAT;
      break;
    case DT_COMPLEX128:
      out_type = DT_DOUBLE;
      break;
    default:
      OP_LOGE("ComplexAbs", "Invalid input dtype: %s", DTypeStr(x_type).c_str());
      return GRAPH_FAILED;
  }
  x_desc.SetDataType(out_type);
  return op.UpdateOutputDesc("y", x_desc);
}

INFER_FUNC_REG(ComplexAbs, ComplexAbsInfer);
// ----------------ComplexAbs End-------------------

// ----------------Complex-------------------
IMPLEMT_INFERFUNC(Complex, ComplexInfer) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "real", "imag", "out", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  TensorDesc x_desc = op.GetInputDescByName("real");
  DataType x_type = x_desc.GetDataType();
  DataType out_type;
  switch (x_type) {
    case DT_FLOAT:
      out_type = DT_COMPLEX64;
      break;
    case DT_DOUBLE:
      out_type = DT_COMPLEX128;
      break;
    default:
      OP_LOGE("Complex", "Invalid input dtype: %s", DTypeStr(x_type).c_str());
      return GRAPH_FAILED;
  }
  TensorDesc out_desc = op.GetOutputDescByName("out");
  out_desc.SetDataType(out_type);
  return op.UpdateOutputDesc("out", out_desc);
}
INFER_FUNC_REG(Complex, ComplexInfer);
// ----------------Complex End-------------------

// ----------------IsNan-------------------
IMPLEMT_INFERFUNC(IsNan, IsNanInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsNan, IsNanInfer);
// ----------------IsNan End-------------------

// ----------------NextAfter-------------------
IMPLEMT_INFERFUNC(NextAfter, NextAfterInfer) {
  Shape x_shape = op.GetInputDescByName("x1").GetShape();
  Shape y_shape = op.GetInputDescByName("x2").GetShape();
  TensorDesc out_desc = op.GetOutputDescByName("output");
  DataType x_type = op.GetInputDescByName("x1").GetDataType();
  DataType y_type = op.GetInputDescByName("x2").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of x1 is different from that of x2!");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(x_type);
  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_SHAPE);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "update output failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  const size_t rank_x = x_shape.GetDimNum();
  const size_t rank_y = y_shape.GetDimNum();
  const size_t rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, zip together x_shape and y_shape
  // and pad with 1 to make them the same length.
  std::vector<int64_t> dims;
  int64_t dim_one = 1;
  if (rank_x != rank_y) {
    OP_LOGI(TbeGetName(op).c_str(), "x1 shape is not equal to x2 shape!");
    dim_one = 1;
  }
  for (size_t i = 0; i < rank_out; ++i) {
    int64_t dim_x;
    if (i < (rank_out - rank_x)) {
      dim_x = dim_one;
    } else {
      // rank_out = rank_x or i >= rank_y - rank_x.
      for (size_t j = 0; j < x_shape.GetDimNum(); ++j) {
        if (x_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_x = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_x)) < 0) {
        dim_x = x_shape.GetDim(rank_x + i - (rank_out - rank_x));
      } else {
        dim_x = x_shape.GetDim(i - (rank_out - rank_x));
      }
    }

    const bool dim_y_is_one = (i < (rank_out - rank_y));
    int64_t dim_y;
    if (dim_y_is_one) {
      dim_y = dim_one;
    } else {
      // rank_out = rank_y or i >= rank_x - rank_y.
      for (size_t j = 0; j < y_shape.GetDimNum(); ++j) {
        if (y_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_y = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_y)) < 0) {
        dim_y = y_shape.GetDim(rank_y + i - (rank_out - rank_y));
      } else {
        dim_y = y_shape.GetDim(i - (rank_out - rank_y));
      }
    }

    if ((dim_x == UNKNOWN_DIM) || (dim_y == UNKNOWN_DIM)) {
      /* One or both dimensions is unknown.
       * If either dimension is greater than 1, assume that the program is
       * correct, and the other dimension will be broadcast to match it.
       * For shape inference, if eliminate the shape checks
       * in this code, assert that the unknown dim is either 1
       * or the same as the known dim.
       * If either dimension is 1, the other dimension is the output.
       */
      if (dim_x > 1) {
        dims.push_back(dim_x);
      } else if (dim_y > 1) {
        dims.push_back(dim_y);
      } else if (dim_x == 1) {
        dims.push_back(dim_y);
      } else if (dim_y == 1) {
        dims.push_back(dim_x);
      } else if (dim_x == dim_y) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(UNKNOWN_DIM);
      }
    } else if ((dim_x == 1) || (dim_y == 1)) {
      // dim_x is dim_one or dim_y is dim_one.
      if ((dim_x == 1) && (!dim_y_is_one)) {
        // broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        if (dim_y == 1) {
          // broadcast dim_y to dim_x.
          dims.push_back(dim_x);
        }
      }
    } else {
      int64_t dim;
      if (Merge(dim_x, dim_y, dim) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
      dims.push_back(dim);
    }
  }
  Shape out_shape(dims);
  out_desc.SetShape(out_shape);
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(NextAfter, NextAfterInfer);
// ----------------NextAfter End-------------------

// ----------------IsInf------------------------
IMPLEMT_INFERFUNC(IsInf, IsInfInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsInf, IsInfInfer);
// ----------------IsInf END------------------------

// ----------------ReduceOp-------------------
static bool InferReduceShapeProcess(Operator op, const int64_t input_x_idx, const int64_t output_y_idx,
                                    const std::string &input_axes_name) {
  bool keep_dims = false;
  op.GetAttr("keep_dims", keep_dims);
  reduce_ops::CommonReduceInferWithInputAxes(op, input_x_idx, output_y_idx, input_axes_name, keep_dims);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(TypicalReduceInferShape) {
  OP_LOGD(TbeGetName(op), "Enter %s InferShape", TbeGetOpType(op).c_str());
  const int64_t input_x_idx = 0;
  const int64_t output_y_idx = 0;
  if (InferReduceShapeProcess(op, input_x_idx, output_y_idx, "axes")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

IMPLEMT_COMMON_INFERFUNC(ScalarReduceProdInferShape) {
  OP_LOGD(TbeGetName(op), "Enter %s InferShape", TbeGetOpType(op).c_str());
  auto input_desc = op.GetInputDesc(0);
  auto output_desc = op.GetOutputDesc(0);
  if (input_desc.GetShape().GetDimNum() == 0) {
    std::vector<int64_t> output_shape{1};
    output_desc.SetShape(Shape(output_shape));
    output_desc.SetDataType(op.GetInputDesc(0).GetDataType());
    op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }
  auto axes_desc = op.GetOutputDesc(1);
  uint8_t *data = nullptr;
  size_t len = 0;
  auto ret = axes_desc.GetConstData(&data, len);
  bool keep_dims = false;
  op.GetAttr("keep_dims", keep_dims);
  if (ret == GRAPH_SUCCESS) {
    const int64_t input_x_idx = 0;
    const int64_t output_y_idx = 0;
    if (reduce_ops::CommonReduceInferWithInputAxes(op, input_x_idx, output_y_idx, "axes", keep_dims)) {
      return GRAPH_SUCCESS;
    }
  }
  const Shape &axes_shape = axes_desc.GetShape();
  if (reduce_ops::DoReduceInferShapeWithoutAxes(op, input_desc, output_desc, axes_shape, keep_dims)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceSum, TypicalReduceInferShape);
COMMON_INFER_FUNC_REG(ReduceProd, ScalarReduceProdInferShape);
// ----------------ReduceOp END-------------------

// ----------------RaggedRange-------------------
IMPLEMT_INFERFUNC(RaggedRange, RaggedRangeInfer) {
  Shape starts;
  Shape limits;
  Shape deltas;
  if (WithRankAtMost(op.GetInputDesc(0), 1, starts, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, ", "input[starts] rank must be at most 1D, got rank[",
                   op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(1), 1, limits, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, ", "input[limits] rank must be at most 1D, got rank[",
                   op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, deltas, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, input[deltas] ", "rank must be at most 1D, got rank[",
                   op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t dim = ge::UNKNOWN_DIM;
  int64_t starts_dim = starts.GetDim(0);
  int64_t limits_dim = limits.GetDim(0);
  int64_t deltas_dim = deltas.GetDim(0);
  if (op.GetInputDesc(0).GetShape().GetDimNum() == 1) {
    if (Merge(starts_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", starts_dim,
                                         "] of input[starts] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(1).GetShape().GetDimNum() == 1) {
    if (Merge(limits_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", limits_dim,
                                         "] of input[limits] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(2).GetShape().GetDimNum() == 1) {
    if (Merge(deltas_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", deltas_dim,
                                         "] of input[deltas] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int64_t rt_nested_splits_dim = ge::UNKNOWN_DIM;
  if (dim != ge::UNKNOWN_DIM) {
    rt_nested_splits_dim = dim + 1;
  } else if (op.GetInputDesc(0).GetShape().GetDimNum() == 0 && op.GetInputDesc(1).GetShape().GetDimNum() == 0 &&
             op.GetInputDesc(2).GetShape().GetDimNum() == 0) {
    rt_nested_splits_dim = 2;
  }

  DataType Tsplits_type;
  if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[Tsplits] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc rt_nested_desc = op.GetOutputDescByName("rt_nested_splits");
  rt_nested_desc.SetShape(Shape({rt_nested_splits_dim}));
  rt_nested_desc.SetDataType(Tsplits_type);
  (void)op.UpdateOutputDesc("rt_nested_splits", rt_nested_desc);

  DataType T_type = op.GetInputDescByName("starts").GetDataType();
  std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
  TensorDesc dense_desc = op.GetOutputDescByName("rt_dense_values");
  dense_desc.SetShape(Shape(unknow_dim_vec));
  dense_desc.SetDataType(T_type);
  (void)op.UpdateOutputDesc("rt_dense_values", dense_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedRange, RaggedRangeInfer);
// ----------------RaggedRange END-------------------

// ----------------Correlate-------------------
CUST_IMPLEMT_INFERFUNC(Correlate, CorrelateInfer) {
  TensorDesc output_desc = op.GetOutputDescByName("output");
  // infer type
  DataType a_type = op.GetInputDescByName("a").GetDataType();
  DataType v_type = op.GetInputDescByName("v").GetDataType();
  if (a_type != v_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of a is different from that of v!");
    return GRAPH_FAILED;
  }
  DataType out_type;

  static const std::vector<DataType> type_to_float32 = {DT_INT16,  DT_INT32,  DT_INT8, DT_BOOL,
                                                        DT_UINT16, DT_UINT32, DT_UINT8};
  static const std::vector<DataType> type_to_float64 = {DT_INT64, DT_UINT64};
  bool is_type_to_float32 = std::any_of(type_to_float32.begin(), type_to_float32.end(),
                                        [&a_type](const DataType &dtype) { return a_type == dtype; });
  bool is_type_to_float64 = std::any_of(type_to_float64.begin(), type_to_float64.end(),
                                        [&a_type](const DataType &dtype) { return a_type == dtype; });
  if (is_type_to_float32)
    out_type = DT_FLOAT;
  else if (is_type_to_float64)
    out_type = DT_DOUBLE;
  else
    out_type = a_type;
  output_desc.SetDataType(out_type);
  // infer shape
  Shape a_shape = op.GetInputDescByName("a").GetShape();
  Shape v_shape = op.GetInputDescByName("v").GetShape();
  if (IsUnknownShape(a_shape) || IsUnknownShape(v_shape)) {
    std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
    output_desc.SetShape(Shape(unknow_dim_vec));
  } else {
    auto a_dimension = a_shape.GetDimNum();
    auto v_dimension = v_shape.GetDimNum();
    if ((a_dimension != 1) || (v_dimension != 1)) {
      string error_msg = ConcatString("'Correlate' only support 1-dimensional inputs , but got a at ", a_dimension,
                                      "-dimensional and got v at ", v_dimension, "-dimensional");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    int64_t a_size = a_shape.GetDim(0);
    int64_t v_size = v_shape.GetDim(0);
    if (a_shape.GetDim(0) == 0 || v_shape.GetDim(0) == 0) {
      string error_msg =
        ConcatString("all inputs of 'Correlate' cannot be empty , got a at (", a_size, ") and got v at (", v_size, ")");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    int64_t out_size;
    int64_t long_size = std::max(a_size, v_size);
    int64_t short_size = std::min(a_size, v_size);
    std::string mode;
    if (op.GetAttr("mode", mode) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), std::string("get attr[mode] failed"));
      return GRAPH_FAILED;
    }
    if (mode == "valid") {
      out_size = long_size - short_size + 1;
    } else if (mode == "same") {
      out_size = long_size;
    } else if (mode == "full") {
      out_size = long_size + short_size - 1;
    } else {
      string error_msg =
        ConcatString("the mode of 'Correlate' should be one of [valid, same, full], but got ", mode, ".");
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), error_msg);
      return GRAPH_FAILED;
    }
    std::vector<int64_t> out_dim_vec(1, out_size);
    output_desc.SetShape(Shape(out_dim_vec));
  }

  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(Correlate, CorrelateInfer);
// ----------------Correlate END-------------------

// ----------------FFTWithSize-------------------
graphStatus FFTWithSizeInferType(const Operator &op, TensorDesc *y_desc, DataType input_type, bool real, bool inverse) {
  enum class FFTType { RFFT, FFT, IRFFT };
  static const std::unordered_map<DataType, DataType> kRfftTypes{
    {DT_FLOAT, DT_COMPLEX64}, {DT_DOUBLE, DT_COMPLEX128}, {DT_UINT8, DT_COMPLEX64}, {DT_INT8, DT_COMPLEX64},
    {DT_INT16, DT_COMPLEX64}, {DT_INT32, DT_COMPLEX64},   {DT_INT64, DT_COMPLEX64}, {DT_BOOL, DT_COMPLEX64}};
  static const std::unordered_map<DataType, DataType> kFftTypes{{DT_COMPLEX64, DT_COMPLEX64},
                                                                {DT_COMPLEX128, DT_COMPLEX128}};
  static const std::unordered_map<DataType, DataType> kIrfftTypes{{DT_COMPLEX64, DT_FLOAT}, {DT_COMPLEX128, DT_DOUBLE}};
  static const std::unordered_map<FFTType, std::unordered_map<DataType, DataType>> kTypeMap{
    {FFTType::RFFT, kRfftTypes}, {FFTType::FFT, kFftTypes}, {FFTType::IRFFT, kIrfftTypes}};
  FFTType fft_type;
  if (real) {
    if (!inverse) {
      fft_type = FFTType::RFFT;
    } else {
      fft_type = FFTType::IRFFT;
    }
  } else {
    fft_type = FFTType::FFT;
  }
  auto &type_map = kTypeMap.at(fft_type);
  if (type_map.find(input_type) == type_map.end()) {
    OP_LOGE(TbeGetName(op), "Infer data type failed.");
    return GRAPH_FAILED;
  }
  auto out_type = type_map.at(input_type);
  y_desc->SetDataType(out_type);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(FFTWithSize, FFTWithSizeInfer) {
  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape().GetDims();
  auto x_type = x_desc.GetDataType();
  auto y_shape = x_shape;

  int64_t signal_ndim;
  bool inverse;
  bool real;
  bool onesided;
  std::vector<int64_t> signal_sizes;
  RETURN_IF_FAILURE(op.GetAttr("inverse", inverse));
  RETURN_IF_FAILURE(op.GetAttr("signal_ndim", signal_ndim));
  RETURN_IF_FAILURE(op.GetAttr("onesided", onesided));
  RETURN_IF_FAILURE(op.GetAttr("real", real));
  // acl would fail to set attr if signal_sizes is empty
  if (op.GetAttr("signal_sizes", signal_sizes) == GRAPH_FAILED) {
    signal_sizes = {};
  }

  if (x_shape.empty()) {
    OP_LOGE(TbeGetName(op).c_str(), "x has an empty shape.");
    return GRAPH_FAILED;
  }

  constexpr int64_t kDimNum = 2;
  if (real && onesided) {
    if (!inverse) {
      y_shape.back() = x_shape.back() / kDimNum + 1;
    } else {
      if (signal_sizes.size() == 0) {
        y_shape.back() = x_shape.back();
      } else {
        y_shape.back() = signal_sizes.back();
      }
    }
  }

  auto y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(y_shape));
  FFTWithSizeInferType(op, &y_desc, x_type, real, inverse);
  return op.UpdateOutputDesc("y", y_desc);
}

CUST_INFER_FUNC_REG(FFTWithSize, FFTWithSizeInfer);
// ----------------ReduceOp END-------------------

// ----------------Trilindices-------------------
IMPLEMT_COMMON_INFERFUNC(TrilindicesInfer) {
  int64_t row;
  int64_t col;
  int64_t offset;
  DataType dtype;
  RETURN_IF_FAILURE(op.GetAttr("row", row));
  RETURN_IF_FAILURE(op.GetAttr("col", col));
  RETURN_IF_FAILURE(op.GetAttr("offset", offset));
  RETURN_IF_FAILURE(op.GetAttr("dtype", dtype));
  int64_t tril_size = GetTrilSize(row, col, offset);
  auto output_desc = op.GetOutputDescByName("output");
  output_desc.SetShape(Shape({2, tril_size}));
  output_desc.SetDataType(dtype);
  return op.UpdateOutputDesc("output", output_desc);
}

CUST_COMMON_INFER_FUNC_REG(TrilIndices, TrilindicesInfer);
// ----------------Trilindices End-------------------

// ----------------Triuindices-------------------
IMPLEMT_COMMON_INFERFUNC(TriuindicesInfer) {
  int64_t row;
  int64_t col;
  int64_t offset;
  DataType dtype;
  RETURN_IF_FAILURE(op.GetAttr("row", row));
  RETURN_IF_FAILURE(op.GetAttr("col", col));
  RETURN_IF_FAILURE(op.GetAttr("offset", offset));
  RETURN_IF_FAILURE(op.GetAttr("dtype", dtype));
  auto output_desc = op.GetOutputDescByName("output");
  auto triu_size = row * col - GetTrilSize(row, col, offset - 1);
  output_desc.SetShape(Shape({2, triu_size}));
  output_desc.SetDataType(dtype);
  return op.UpdateOutputDesc("output", output_desc);
}

CUST_COMMON_INFER_FUNC_REG(TriuIndices, TriuindicesInfer);
// ----------------Triuindices End-------------------

// -----------------------CholeskyInverse---------------------
IMPLEMT_COMMON_INFERFUNC(CholeskyInverseInferShape) {
  TensorDesc out_desc = op.GetInputDescByName("x");
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(CholeskyInverse, CholeskyInverseInferShape);

// -----------------------CholeskyInverse END---------------------

// ----------------TraceV2 Begin------------------------
CUST_IMPLEMT_INFERFUNC(TraceV2, TraceV2InferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("output");
  // infer type
  DataType input_type = op.GetInputDescByName("input").GetDataType();
  ge::DataType output_dtype;
  op.GetAttr("dtype", output_dtype);
  if (op.GetAttr("dtype", output_dtype) == GRAPH_SUCCESS && output_dtype != DT_UNDEFINED) {
    output_desc.SetDataType(output_dtype);
  } else {
    OP_LOGW(TbeGetName(op).c_str(), "get attr dtype failed.");
    static const std::vector<DataType> type_to_int64 = {DT_INT16, DT_INT32, DT_INT8, DT_UINT16, DT_UINT32, DT_UINT8};
    bool is_type_to_int64 = std::any_of(type_to_int64.begin(), type_to_int64.end(),
                                        [&input_type](const DataType &dtype) { return input_type == dtype; });
    if (is_type_to_int64) {
      output_desc.SetDataType(DT_INT64);
    } else {
      output_desc.SetDataType(input_type);
    }
  }

  // infer shape
  Shape input_shape = op.GetInputDescByName("input").GetShape();
  if (IsUnknownDimNum(input_shape)) {
    output_desc.SetShape(input_shape);
  } else {
    Tensor axis1_data;
    DataType axis1_dtype = op.GetInputDescByName("axis1").GetDataType();
    if (op.GetInputConstData("axis1", axis1_data) == GRAPH_FAILED) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), ConcatString("get const data from input[axis1] failed in [TraceV2InferShape]"));
      return GRAPH_FAILED;
    }
    std::vector<int64_t> axis1_value;
    GetConstValue(op, axis1_data, axis1_dtype, axis1_value);
    Tensor axis2_data;
    DataType axis2_dtype = op.GetInputDescByName("axis2").GetDataType();
    if (op.GetInputConstData("axis2", axis2_data) == GRAPH_FAILED) {
      AICPU_INFER_SHAPE_INNER_ERR_REPORT(
        TbeGetName(op), ConcatString("get const data from input[axis1] failed in [TraceV2InferShape]"));
      return GRAPH_FAILED;
    }
    std::vector<int64_t> axis2_value;
    GetConstValue(op, axis2_data, axis2_dtype, axis2_value);
    std::vector<int64_t> input_shape_vec = input_shape.GetDims();
    int64_t axis1 = axis1_value[0];
    int64_t axis2 = axis2_value[0];
    int64_t input_rank = input_shape_vec.size();
    axis1 = axis1 < 0 ? axis1 + input_rank : axis1;
    axis2 = axis2 < 0 ? axis2 + input_rank : axis2;
    std::vector<int64_t> output_shape_vec;
    for (int64_t i = 0; i < input_rank; i++) {
      if (i != axis1 && i != axis2) {
        output_shape_vec.emplace_back(input_shape_vec[i]);
      }
    }
    output_desc.SetShape(Shape(output_shape_vec));
  }
  if (op.UpdateOutputDesc("output", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(TraceV2, TraceV2InferShape);
// ---------------TraceV2 END-------------------------------

// ----------------TraceV2Grad Begin------------------------
CUST_IMPLEMT_INFERFUNC(TraceV2Grad, TraceV2GradInferShape) {
  TensorDesc din_desc = op.GetOutputDescByName("din");
  // infer type
  DataType dout_type = op.GetInputDescByName("dout").GetDataType();
  din_desc.SetDataType(dout_type);

  // infer shape
  DataType shape_dtype = op.GetInputDescByName("shape").GetDataType();
  Tensor shape_data;
  std::vector<int64_t> shape_value;
  if (op.GetInputConstData("shape", shape_data) == GRAPH_FAILED) {
    shape_value = {-2};
  } else {
    GetConstValue(op, shape_data, shape_dtype, shape_value);
  }
  din_desc.SetShape(Shape(shape_value));
  if (op.UpdateOutputDesc("din", din_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update din desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(TraceV2Grad, TraceV2GradInferShape);
// ---------------TraceV2Grad END-------------------------------

// ----------------Logit-------------------
CUST_IMPLEMT_INFERFUNC(Logit, LogitInfer) {
  TensorDesc out_desc = op.GetInputDescByName("x");
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(Logit, LogitInfer);
// ----------------Logit END-------------------

// ----------------Diagonal-------------------
CUST_IMPLEMT_INFERFUNC(Diagonal, DiagonalInfer) {
  int64_t offset;
  int64_t dim1;
  int64_t dim2;
  RETURN_IF_FAILURE(op.GetAttr("offset", offset));
  RETURN_IF_FAILURE(op.GetAttr("dim1", dim1));
  RETURN_IF_FAILURE(op.GetAttr("dim2", dim2));
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_type = op.GetInputDescByName("x").GetDataType();

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(x_type);
  if ((!RankKnown(x_shape))) {
    Shape y_shape(UNKNOWN_RANK);
    y_desc.SetShape(y_shape);
    if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "update y failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  const int64_t min_rank = 2;
  int64_t x_rank = static_cast<int64_t>(x_shape.GetDimNum());
  if (x_rank < min_rank) {
    OP_LOGE(TbeGetName(op).c_str(), "rank of x is less than 2");
    return GRAPH_FAILED;
  }
  if (dim1 < -x_rank || dim1 > x_rank - 1) {
    OP_LOGE(TbeGetName(op).c_str(), "dim1 is not in [-x_rank, x_rank - 1]");
    return GRAPH_FAILED;
  }
  if (dim2 < -x_rank || dim2 > x_rank - 1) {
    OP_LOGE(TbeGetName(op).c_str(), "dim2 is not in [-x_rank, x_rank - 1]");
    return GRAPH_FAILED;
  }
  int64_t tmp_dim1 = dim1 < 0 ? dim1 + x_rank : dim1;
  int64_t tmp_dim2 = dim2 < 0 ? dim2 + x_rank : dim2;
  if (tmp_dim1 == tmp_dim2) {
    OP_LOGE(TbeGetName(op).c_str(), "dim1 can not be equal to dim2");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> x_shape_vec = x_shape.GetDims();
  std::vector<int64_t> y_shape_vec;
  for (int64_t dim = 0; dim < x_rank; dim++) {
    if (dim != tmp_dim1 && dim != tmp_dim2) {
      y_shape_vec.emplace_back(x_shape_vec[dim]);
    }
  }
  int64_t dsize = UNKNOWN_DIM;
  if (x_shape_vec[tmp_dim1] != UNKNOWN_DIM && x_shape_vec[tmp_dim2] != UNKNOWN_DIM) {
    if (offset >= 0) {
      dsize = std::max<int64_t>(std::min(x_shape_vec[tmp_dim1], x_shape_vec[tmp_dim2] - offset), 0);
    } else {
      dsize = std::max<int64_t>(std::min(x_shape_vec[tmp_dim1] + offset, x_shape_vec[tmp_dim2]), 0);
    }
  }
  y_shape_vec.emplace_back(dsize);

  Shape y_shape(y_shape_vec);
  y_desc.SetShape(y_shape);
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update y failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(Diagonal, DiagonalInfer);
// ----------------Diagonal END-------------------
}  // namespace ge
