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

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "common/mockcpp.h"
#include "pynative/common.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_def.h"
#include "pynative/forward/pyboost/converter.h"
#include "pynative/utils/pynative_utils.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace pynative {
class PyBoostConverterTest : public PyCommon {};

/// Feature: Test Pyboost Converter.
/// Description: Test ToBasicInt for pyboost input converter.
/// Expectation: ToBasicInt will throw exception when input is invalid.
TEST_F(PyBoostConverterTest, ToBasicInt_TypeCastError) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("invalid"));
  converter.Parse(list.ptr());

  EXPECT_THROW({
    converter.ToBasicInt(list.ptr(), kIndex0);
  }, std::exception);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBasicIntOptional for pyboost input converter.
/// Expectation: ToBasicIntOptional success.
TEST_F(PyBoostConverterTest, ToBasicIntOptionalTest) {
  Converter converter(&ops::gTriu);

  auto x_obj = py::none();
  auto y_obj = NewPyTensor(tensor::from_scalar(1));

  py::list list;
  list.append(x_obj);
  list.append(y_obj);
  converter.Parse(list.ptr());

  auto x_out = converter.ToBasicIntOptional(list.ptr(), kIndex0);
  auto y_out = converter.ToBasicIntOptional(list.ptr(), kIndex1);

  ASSERT_EQ(x_out.has_value(), false);
  ASSERT_EQ(y_out.value(), 1);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBasicIntVectorOptional for pyboost input converter.
/// Expectation: ToBasicIntVectorOptional success.
TEST_F(PyBoostConverterTest, ToBasicIntVectorOptionalTest) {
  Converter converter(&ops::gResizeNearestNeighborV2Grad);

  py::list python_args;
  python_args.append(py::none());
  python_args.append(NewPyTensor(tensor::from_vector(std::vector<int>{1, 2, 3})));
  python_args.append(py::none());
  python_args.append(py::none());
  converter.Parse(python_args.ptr());

  auto x_out = converter.ToBasicIntVectorOptional(python_args.ptr(), kIndex0);
  auto y_out = converter.ToBasicIntVectorOptional(python_args.ptr(), kIndex1);

  ASSERT_EQ(x_out.has_value(), false);
  ASSERT_EQ(y_out.has_value(), true);
  ASSERT_EQ(y_out.value(), (std::vector<int64_t>{1, 2, 3}));
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensor for pyboost input converter.
/// Expectation: Python Tensor to Tensor success.
TEST_F(PyBoostConverterTest, ToTensorTest1) {
  Converter converter(&ops::gSin);

  auto tensor_py = NewPyTensor(tensor::from_scalar(1));

  py::list list;
  list.append(tensor_py);
  converter.Parse(list.ptr());

  auto t = converter.ToTensor(list.ptr(), kIndex0);
  ASSERT_EQ(t, tensor::ConvertToTensor(tensor_py));
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensor for pyboost input converter.
/// Expectation: Python float to Tensor success.
TEST_F(PyBoostConverterTest, ToTensorTest3) {
  Converter converter(&ops::gAdd);

  auto x_obj = NewPyTensor(tensor::from_scalar(1));
  auto y_obj = py::float_(1.0);

  py::list list;
  list.append(x_obj);
  list.append(y_obj);
  converter.Parse(list.ptr());

  auto x_out = converter.ToTensor(list.ptr(), kIndex0);
  auto y_out = converter.ToTensor(list.ptr(), kIndex1);
  ASSERT_NE(x_out, nullptr);
  ASSERT_NE(y_out, nullptr);
  ASSERT_EQ(y_out->isa<tensor::Tensor>(), true);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensorOptional for pyboost input input converter.
/// Expectation: ToTensorOptional return none when input is py::none().
TEST_F(PyBoostConverterTest, ToTensorOptionalTest) {
  Converter converter(&ops::gClampTensor);

  auto input = NewPyTensor(tensor::from_scalar(1));
  auto min = NewPyTensor(tensor::from_scalar(1));
  auto max = py::none();

  py::list list;
  list.append(input);
  list.append(min);
  list.append(max);
  converter.Parse(list.ptr());

  auto min_out = converter.ToTensorOptional(list.ptr(), kIndex1);
  ASSERT_EQ(min_out.has_value(), true);
  ASSERT_NE(min_out.value(), nullptr);
  ASSERT_EQ(min_out.value()->isa<tensor::Tensor>(), true);

  auto max_out = converter.ToTensorOptional(list.ptr(), kIndex2);
  ASSERT_EQ(max_out.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensorOptional for pyboost input converter.
/// Expectation: To int success.
TEST_F(PyBoostConverterTest, ToIntOptionalTest1) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gArgMaxExt);

  auto input = NewPyTensor(tensor::from_scalar(1));
  auto dim = py::none();
  auto keep_dim = py::bool_(true);

  py::list list;
  list.append(input);
  list.append(dim);
  list.append(keep_dim);
  converter.Parse(list.ptr());

  auto input_out = converter.ToTensor(list.ptr(), kIndex0);
  ASSERT_NE(input_out, nullptr);
  ASSERT_EQ(input_out->isa<tensor::Tensor>(), true);

  auto dim_out = converter.ToIntOptional(list.ptr(), kIndex1);
  ASSERT_EQ(dim_out.has_value(), false);

  auto keep_dim_out = converter.ToBool(list.ptr(), kIndex2);
  ASSERT_EQ(keep_dim_out->value(), true);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToIntOptional for pyboost input converter.
/// Expectation: To in success.
TEST_F(PyBoostConverterTest, ToIntOptionalTest2) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gArgMaxExt);

  auto input = NewPyTensor(tensor::from_scalar(1));
  auto dim = py::int_(1);
  auto keep_dim = py::bool_(false);

  py::list list;
  list.append(input);
  list.append(dim);
  list.append(keep_dim);
  converter.Parse(list.ptr());

  auto dim_out = converter.ToIntOptional(list.ptr(), kIndex1);
  ASSERT_EQ(dim_out.has_value(), true);
  ASSERT_EQ(dim_out.value()->value(), 1);

  auto keep_dim_out = converter.ToBool(list.ptr(), kIndex2);
  ASSERT_EQ(keep_dim_out->value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToInt for pyboost input converter.
/// Expectation: ToInt will throw exception when input is invalid.
TEST_F(PyBoostConverterTest, ToInt_TypeCastError) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("invalid"));
  converter.Parse(list.ptr());

  EXPECT_THROW({
    converter.ToInt(list.ptr(), kIndex0);
  }, std::exception);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoolOptional for pyboost input converter.
/// Expectation: To bool and get none.
TEST_F(PyBoostConverterTest, ToBoolOptionalTest1) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::none());
  converter.Parse(list.ptr());

  auto t = converter.ToBoolOptional(list.ptr(), kIndex0);
  ASSERT_EQ(t.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoolOptional for pyboost input converter.
/// Expectation: To bool success.
TEST_F(PyBoostConverterTest, ToBoolOptionalTest2) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::bool_(true));
  converter.Parse(list.ptr());

  auto t = converter.ToBoolOptional(list.ptr(), kIndex0);

  ASSERT_EQ(t.has_value(), true);
  ASSERT_EQ(t.value()->value(), true);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoollist for pyboost input converter.
/// Expectation: ToBoollist success.
TEST_F(PyBoostConverterTest, ToBoolListTest) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list bool_list;
  bool_list.append(py::bool_(false));
  bool_list.append(py::bool_(true));
  py::list python_args;
  python_args.append(bool_list);
  converter.Parse(python_args.ptr());

  auto result = converter.ToBoolList<CPythonList>(python_args.ptr(), kIndex0);

  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->size(), 2);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoollistOptional for pyboost input converter.
/// Expectation: ToBoollistOptional success.
TEST_F(PyBoostConverterTest, ToBoolListOptionalTest) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gAdd);

  py::list bool_list_1;
  bool_list_1.append(py::bool_(false));
  bool_list_1.append(py::bool_(false));

  py::list python_args;
  python_args.append(bool_list_1);
  python_args.append(py::none());
  converter.Parse(python_args.ptr());

  auto result_1 = converter.ToBoolListOptional<CPythonList>(python_args.ptr(), kIndex0);
  auto result_2 = converter.ToBoolListOptional<CPythonList>(python_args.ptr(), kIndex1);

  ASSERT_EQ(result_1.has_value(), true);

  for(auto val : result_1.value()->value()) {
    auto bool_imm = std::dynamic_pointer_cast<BoolImm>(val);
    ASSERT_EQ(bool_imm->value(), false);
  }

  ASSERT_EQ(result_2.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToFloat for pyboost input converter.
/// Expectation: ToFloat will throw exception when input is invalid.
TEST_F(PyBoostConverterTest, ToFloat_TypeCastError) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("invalid"));
  converter.Parse(list.ptr());

  EXPECT_THROW({
    converter.ToFloat(list.ptr(), kIndex0);
  }, std::exception);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToScalar for pyboost input converter.
/// Expectation: ToScalar will throw exception when input is invalid.
TEST_F(PyBoostConverterTest, ToScalar_TypeCastError) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("invalid"));
  converter.Parse(list.ptr());

  EXPECT_THROW({
    converter.ToScalar(list.ptr(), kIndex0);
  }, std::exception);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToString for pyboost input converter.
/// Expectation: ToString success.
TEST_F(PyBoostConverterTest, ToStringTest) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("valid_string"));
  converter.Parse(list.ptr());

  auto result = converter.ToString(list.ptr(), kIndex0);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->value(), "valid_string");
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToStringOptional with none input.
/// Expectation: Return none when input is py::none().
TEST_F(PyBoostConverterTest, ToStringOptionalTest1) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::none());
  converter.Parse(list.ptr());

  auto result = converter.ToStringOptional(list.ptr(), kIndex0);
  ASSERT_EQ(result.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToStringOptional with valid string input.
/// Expectation: ToStringOptional success.
TEST_F(PyBoostConverterTest, ToStringOptionalTest2) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("valid_string"));
  converter.Parse(list.ptr());

  auto result = converter.ToStringOptional(list.ptr(), kIndex0);
  ASSERT_EQ(result.has_value(), true);
  ASSERT_EQ(result.value()->value(), "valid_string");
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToDtype for pyboost input converter.
/// Expectation: ToDtype will throw exception when input is invalid.
TEST_F(PyBoostConverterTest, ToDtype_TypeCastError) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::str("invalid"));
  converter.Parse(list.ptr());

  EXPECT_THROW({
    converter.ToDtype(list.ptr(), kIndex0);
  }, std::exception);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ClearArgs and PrintConvertError.
/// Expectation: ClearArgs and PrintConvertError success.
TEST_F(PyBoostConverterTest, ParserArgsStructTest) {
  py::gil_scoped_acquire gil;

  auto signature = std::make_shared<FunctionSignature>("Add(int|tensor x)", 0, "test_signature");

  FunctionParameter func_para("int|tensor x", 0);

  signature->params_ = std::vector<FunctionParameter>{func_para};

  ParserArgs Pa_arg(signature);

  auto python_args = py::int_(1);
  auto convert_type = ConvertPair(ops::DT_INT, ops::DT_TENSOR);
  Pa_arg.SetArg(python_args.ptr(), convert_type, kIndex0);

  Pa_arg.ClearArgs();

  ASSERT_EQ(Pa_arg.arg_list_.size(), 0);
  ASSERT_EQ(Pa_arg.src_types_.size(), 0);
  ASSERT_EQ(Pa_arg.dst_types_.size(), 0);

  EXPECT_THROW({
    Pa_arg.PrintConvertError(kIndex0);
  }, std::exception);
}
}  // namespace pynative
}  // namespace mindspore