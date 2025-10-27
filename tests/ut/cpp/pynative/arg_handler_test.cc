/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pynative/forward/pyboost/arg_handler.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "include/utils/tensor_py.h"
#include "mindspore/core/include/ir/dtype/type.h"
#include "mindspore/core/include/ir/dtype/type_id.h"
#include "ops/op_def.h"
#include <algorithm>
#include <vector>
#include <tuple>
#include <string>

namespace mindspore {
namespace pynative {

class ArgHandlerTest : public PyCommon {};

/// Feature: Test arg_handler.
/// Description: Test DtypeToTypeId.
/// Expectation: DtypeToTypeId success.
TEST_F(ArgHandlerTest, DtypeToTypeIdTest) {
  py::gil_scoped_acquire gil;

  auto obj_1 = py::none();

  auto type = mindspore::Int(32);  // type is an instance of mindspore::type representing int32 data type
  py::object obj_2 = py::cast(type);

  auto obj_3 = py::type::of(py::bool_());
  auto obj_4 = py::bool_(true);

  auto result_1 = DtypeToTypeId("Add", "x", obj_1);
  ASSERT_EQ(result_1.has_value(), false);

  auto result_2 = DtypeToTypeId("Add", "x", obj_2);
  ASSERT_EQ(result_2.has_value(), true);

  auto result_3 = DtypeToTypeId("Add", "x", obj_3);
  ASSERT_EQ(result_3.has_value(), true);

  EXPECT_THROW(
    {
      auto result_4 =
        DtypeToTypeId("Add", "x", obj_4);  // obj_4 is an instance of Bool type, rather than the Bool type itself.
    },
    std::exception);
}

/// Feature: Test arg_handler.
/// Description: Test StrToEnum.
/// Expectation: StrToEnum success.
TEST_F(ArgHandlerTest, StrToEnumTest) {
  py::gil_scoped_acquire gil;

  auto obj_1 = py::none();
  auto obj_2 = py::str("VALID");
  auto obj_3 = py::int_(1);

  auto result_1 = StrToEnum("Add", "pad_mode", obj_1);
  ASSERT_EQ(result_1.has_value(), false);

  auto result_2 = StrToEnum("Add", "pad_mode", obj_2);
  ASSERT_EQ(result_2.has_value(), true);

  EXPECT_THROW({ auto result_3 = StrToEnum("Add", "pad_mode", obj_3); }, std::exception);
}

/// Feature: Test arg_handler.
/// Description: Test ToPair.
/// Expectation: ToPair success.
TEST_F(ArgHandlerTest, ToPairTest) {
  py::gil_scoped_acquire gil;

  py::object arg_val_1 = py::int_(5);
  auto result_1 = ToPair("TestOp", "shape", arg_val_1);
  EXPECT_EQ(result_1, std::vector<int>({5, 5}));

  py::list lst;
  lst.append(2);
  lst.append(4);
  auto result_2 = ToPair("TestOp", "shape", lst);
  EXPECT_EQ(result_2, std::vector<int>({2, 4}));

  py::tuple tup = py::make_tuple(3, 6);
  auto result_3 = ToPair("TestOp", "kernel_size", tup);
  EXPECT_EQ(result_3, std::vector<int>({3, 6}));

  py::object arg_val = py::str("invalid");
  EXPECT_THROW({ auto result_4 = ToPair("TestOp", "invalid_arg", arg_val); }, std::runtime_error);
}

/// Feature: Test arg_handler.
/// Description: Test To2dPaddings.
/// Expectation: To2dPaddings success.
TEST_F(ArgHandlerTest, To2dPaddingsTest) {
  py::gil_scoped_acquire gil;

  py::object pad_1 = py::int_(3);
  auto result_1 = To2dPaddings("Conv2D", "padding", pad_1);
  EXPECT_EQ(result_1, std::vector<int>({3, 3}));

  py::list lst;
  lst.append(1);
  lst.append(2);
  auto result_2 = To2dPaddings("Pad", "paddings", lst);
  EXPECT_EQ(result_2, std::vector<int>({1, 2}));

  py::list empty_lst;
  auto result_3 = To2dPaddings("CustomOp", "empty_pads", empty_lst);
  EXPECT_TRUE(result_3.empty());
}

/// Feature: Test arg_handler.
/// Description: Test ToKernelSize.
/// Expectation: ToKernelSize success.
TEST_F(ArgHandlerTest, ToKernelSize_SingleInteger) {
  py::object arg = py::int_(3);
  std::vector<int> result = ToKernelSize("Conv2D", "kernel_size", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 3);
}

/// Feature: Test arg_handler.
/// Description: Test ToKernelSize with list.
/// Expectation: ToKernelSize success.
TEST_F(ArgHandlerTest, ToKernelSize_ListOfFour) {
  py::list arg = py::make_tuple(1, 2, 3, 4);
  std::vector<int> result = ToKernelSize("Conv2D", "kernel_size", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
}

/// Feature: Test arg_handler.
/// Description: Test ToStrides.
/// Expectation: ToStrides success.
TEST_F(ArgHandlerTest, ToStrides_SingleInteger) {
  py::object arg = py::int_(2);
  std::vector<int> result = ToStrides("Conv2D", "stride", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[1], 2);
}

/// Feature: Test arg_handler.
/// Description: Test ToStrides with 4-length list.
/// Expectation: ToStrides will use index 2 and 3.
TEST_F(ArgHandlerTest, ToStrides_ListOfFour) {
  py::list arg = py::make_tuple(1, 1, 3, 3);
  std::vector<int> result = ToStrides("Conv2D", "stride", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 3);
}

/// Feature: Test arg_handler.
/// Description: Test ToStrides with 3-length list.
/// Expectation: ToStrides success.
TEST_F(ArgHandlerTest, ToStrides_ListOfThree) {
  py::list arg = py::make_tuple(1, 2, 3);
  std::vector<int> result = ToStrides("Conv2D", "stride", arg);

  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
}

/// Feature: Test arg_handler.
/// Description: Test ToDilations.
/// Expectation: ToDilations success.
TEST_F(ArgHandlerTest, ToDilations_SingleInteger) {
  py::object arg = py::int_(1);
  std::vector<int> result = ToDilations("Conv2D", "dilation", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 1);
}

/// Feature: Test arg_handler.
/// Description: Test ToDilations with list.
/// Expectation: ToDilations success.
TEST_F(ArgHandlerTest, ToDilations_ListOfFour) {
  py::list arg = py::make_tuple(0, 0, 2, 2);
  std::vector<int> result = ToDilations("Conv2D", "dilation", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[1], 2);
}

/// Feature: Test arg_handler.
/// Description: Test ToOutputPadding.
/// Expectation: ToOutputPadding success.
TEST_F(ArgHandlerTest, ToOutputPadding_SingleInteger) {
  py::object arg = py::int_(4);
  std::vector<int> result = ToOutputPadding("Conv2DTranspose", "output_padding", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 4);
  EXPECT_EQ(result[1], 4);
}

/// Feature: Test arg_handler.
/// Description: Test ToOutputPadding with list.
/// Expectation: ToOutputPadding success.
TEST_F(ArgHandlerTest, ToOutputPadding_ListOfFour) {
  py::list arg = py::make_tuple(1, 2, 3, 4);
  std::vector<int> result = ToOutputPadding("Conv2DTranspose", "output_padding", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
}

/// Feature: Test arg_handler.
/// Description: Test ToRates.
/// Expectation: ToRates success.
TEST_F(ArgHandlerTest, ToRates_SingleInteger) {
  py::object arg = py::int_(2);
  std::vector<int> result = ToRates("Dilate", "rates", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 2);
  EXPECT_EQ(result[1], 2);
}

/// Feature: Test arg_handler.
/// Description: Test ToRates with list.
/// Expectation: ToRates success.
TEST_F(ArgHandlerTest, ToRates_ListOfFour) {
  py::list arg = py::make_tuple(0, 0, 3, 4);
  std::vector<int> result = ToRates("Dilate", "rates", arg);

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 3);
  EXPECT_EQ(result[1], 4);
}

/// Feature: Test arg_handler.
/// Description: Test exception.
/// Expectation: Throw exception when input is invalid.
TEST_F(ArgHandlerTest, AllFunctions_ExceptionHandling) {
  py::object invalid_arg = py::str("invalid");

  EXPECT_THROW({ auto result = ToPair("TestOp", "shape", invalid_arg); }, std::exception);

  EXPECT_THROW({ auto result = ToKernelSize("Conv2D", "kernel_size", invalid_arg); }, std::exception);

  EXPECT_THROW({ auto result = ToStrides("Conv2D", "stride", invalid_arg); }, std::exception);

  EXPECT_THROW({ auto result = ToDilations("Conv2D", "dilation", invalid_arg); }, std::exception);

  EXPECT_THROW({ auto result = ToOutputPadding("Conv2DTranspose", "output_padding", invalid_arg); }, std::exception);

  EXPECT_THROW({ auto result = ToRates("Dilate", "rates", invalid_arg); }, std::exception);
}
}  // namespace pynative
}  // namespace mindspore
