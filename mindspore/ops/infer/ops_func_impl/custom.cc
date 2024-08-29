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

#include "infer/ops_func_impl/custom.h"
#include <algorithm>
#include <string>

#if !defined(_WIN32) && !defined(_WIN64) && !defined(BUILD_LITE) && !defined(__APPLE__)
#include <dlfcn.h>
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "pybind_api/ir/primitive_py.h"
#endif

#include "include/common/utils/utils.h"
#include "utils/check_convert_utils.h"
#include "utils/file_utils.h"
#include "utils/custom_aot_extra.h"
#include "mindspore/core/abstract/abstract_value.h"

namespace mindspore::ops {
namespace {
constexpr auto kCppInferShapeAttr = "cpp_infer_shape";
constexpr auto kFuncName = "func_name";
constexpr auto kAOTFuncType = "aot";
constexpr auto kRegOpName = "reg_op_name";

#if !defined(_WIN32) && !defined(_WIN64) && !defined(BUILD_LITE) && !defined(__APPLE__)
std::vector<AbstractBasePtr> GetInputAbstract(const std::string &op_name,
                                              const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() != 1) {
    MS_LOG(EXCEPTION) << "Custom op [" << op_name << "] input args size is not equal 1, but is " << input_args.size();
  }
  auto input = input_args[0];
  if (!input->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Custom op [" << op_name
                      << "] input abstract type is not tuple, abstract: " << input->ToString();
  }
  auto tuple_input = input->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_input);
  return tuple_input->elements();
}
#endif
}  // namespace

BaseShapePtr CustomFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(BUILD_LITE) && !defined(__APPLE__)
  auto op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer shape for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer shape for " << op_type;
  if (!primitive->HasAttr(kCppInferShapeAttr)) {
    if (!primitive->isa<PrimitivePy>()) {
      MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
    }
    auto inputs = GetInputAbstract(op_type, input_args);
    // cppcheck-suppress unreadVariable
    py::gil_scoped_acquire acquire;
    auto prim_py = dyn_cast<PrimitivePy>(primitive);
    auto py_args = PreparePyInputs(inputs);
    auto out_dict = prim_py->RunInfer(py_args);
    auto shape_obj = out_dict[ATTR_SHAPE];
    auto res_vec = shape_obj.cast<ShapeVector>();
    return std::make_shared<abstract::Shape>(res_vec);
  }

  auto func_type = GetValue<std::string>(primitive->GetAttr(kAttrFuncType));
  const auto &exec_info = GetValue<std::string>(primitive->GetAttr(kFuncName));
  if (func_type != kAOTFuncType) {
    MS_LOG(EXCEPTION) << "The custom operator of type '" << func_type
                      << "' does not support shape inference in cpp yet, func name:" << exec_info;
  }

  std::string file_path;
  std::string func_name;
  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For '" << op_type << "', couldn't find the AOT binary file under path: " << path;
    }
    file_path = real_path.value();
    func_name = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_type << "', user defined function path '" << exec_info << "' is illegal.";
  }

  std::vector<int64_t *> input_shapes;
  std::vector<int> ndims;
  std::vector<std::vector<int64_t>> shape_list;
  auto kernel_name = primitive->name();
  auto inputs = GetInputAbstract(op_type, input_args);
  for (size_t idx = 0; idx < inputs.size(); idx++) {
    auto params_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(kernel_name, inputs, idx);
    MS_EXCEPTION_IF_NULL(params_shape_ptr);
    auto params_shape = params_shape_ptr->shape();
    ndims.push_back(SizeToInt(params_shape.size()));
    (void)shape_list.emplace_back(params_shape);
  }
  (void)std::transform(std::begin(shape_list), std::end(shape_list), std::back_inserter(input_shapes),
                       [](auto &v) { return &v[0]; });

  void *handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MS_LOG(EXCEPTION) << "For '" << op_type << "', dlopen file under path" << file_path
                      << "throw the error: " << dlerror();
  }
  AotExtraImpl attrs;
  attrs.SetKernelPrim(primitive);

  auto infer_func = reinterpret_cast<std::add_pointer<std::vector<int64_t>(int *, int64_t **, AotExtra *)>::type>(
    dlsym(handle, (func_name + "InferShape").c_str()));
  if (infer_func == nullptr) {
    MS_LOG(EXCEPTION) << "Get infer shape functions failed. The custom operator does not support dynamic shape yet,"
                      << " func name:" << func_name
                      << ". Add the cpp version of the infer shape function to support dynamic shape.";
  }

  std::vector<int64_t> ret;
  try {
    ret = infer_func(&ndims[0], &input_shapes[0], (&attrs));
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "For " << kernel_name << ", operator failed when executing user defined file " << file_path
                      << "! Error message is " << e.what();
  }

  (void)dlclose(handle);
  attrs.DestructKernelData();
  return std::make_shared<abstract::Shape>(ret);
#else
  MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
  return mindspore::abstract::kNoShape;
#endif
}

TypePtr CustomFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(BUILD_LITE) && !defined(__APPLE__)
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer type for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer type for " << op_type;
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
  }
  auto inputs = GetInputAbstract(op_type, input_args);
  // cppcheck-suppress unreadVariable
  py::gil_scoped_acquire acquire;
  auto prim_py = dyn_cast<PrimitivePy>(primitive);
  auto py_args = PreparePyInputs(inputs);
  auto output = prim_py->RunInfer(py_args);
  auto res_dtype = output[ATTR_DTYPE].cast<TypePtr>();
  if (res_dtype == nullptr) {
    MS_LOG(EXCEPTION)
      << "For custom ops with cpp infer shape functions, we support the case that the output is a tensor."
      << "Thus the inferred dtype should be a type object, but get inferred dtype in: " << output;
  }
  return res_dtype;
#else
  MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
#endif
}
}  // namespace mindspore::ops
