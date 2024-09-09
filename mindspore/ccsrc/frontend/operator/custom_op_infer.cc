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
#if !defined(_WIN32) && !defined(_WIN64)
#include <dlfcn.h>
#endif

#include "include/common/utils/utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "pybind_api/ir/primitive_py.h"
#include "pipeline/jit/ps/static_analysis/prim.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "utils/file_utils.h"
#include "utils/custom_aot_extra.h"
#include "mindspore/ops/infer/custom.h"
#include "infer/ops_func_impl/custom_ext.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kCppInferShapeAttr = "cpp_infer_shape";
constexpr auto kFuncName = "func_name";
constexpr auto kAOTFuncType = "aot";
constexpr auto kRegOpName = "reg_op_name";
constexpr auto kDTypeAttr = "dtype";

#if !defined(_WIN32) && !defined(_WIN64)
std::vector<AbstractBasePtr> GetInputAbstract(const std::string &op_name,
                                              const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.size() != 1) {
    MS_LOG(INFO) << "Custom op [" << op_name << "] input args size is not equal 1, but is " << input_args.size();
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

BaseShapePtr CustomInferShapeByCPP(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto func_type = GetValue<std::string>(primitive->GetAttr(kAttrFuncType));
  const auto &exec_info = GetValue<std::string>(primitive->GetAttr(kFuncName));
  if (func_type != kAOTFuncType) {
    MS_LOG(EXCEPTION) << "The custom operator of type '" << func_type
                      << "' does not support shape inference in cpp yet, func name:" << exec_info;
  }

  auto kernel_name = primitive->name();
  std::string file_path;
  std::string func_name;

  if (auto pos = exec_info.find(":"); pos != std::string::npos) {
    auto path = exec_info.substr(0, pos);
    auto real_path = FileUtils::GetRealPath(path.c_str());
    if (!real_path.has_value()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', couldn't find the AOT binary file under path: " << path;
    }
    file_path = real_path.value();
    func_name = exec_info.substr(pos + 1);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', user defined function path '" << exec_info << "' is illegal.";
  }

  std::vector<int64_t *> input_shapes;
  std::vector<int> ndims;

  std::vector<std::vector<int64_t>> shape_list;

  for (size_t idx = 0; idx < input_args.size(); idx++) {
    auto params_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(kernel_name, input_args, idx);
    MS_EXCEPTION_IF_NULL(params_shape_ptr);
    auto params_shape = params_shape_ptr->shape();
    ndims.push_back(SizeToInt(params_shape.size()));
    (void)shape_list.emplace_back(params_shape);
  }
  (void)std::transform(std::begin(shape_list), std::end(shape_list), std::back_inserter(input_shapes),
                       [](auto &v) { return &v[0]; });

  void *handle = dlopen(file_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', dlopen file under path" << file_path
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
}
#endif
}  // namespace

#define REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(name, primitive, OP_INFER_ClASS, is_impl_infer_value) \
  const auto helper_op_infer_##name = abstract::RegisterStandardPrimitiveEvalHelper(               \
    abstract::GetPrimitiveInferMapPtr(), primitive, std::make_shared<OP_INFER_ClASS>(), is_impl_infer_value);

class AGCustomInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
#if !defined(_WIN32) && !defined(_WIN64)
    return CustomInferShapeByCPP(primitive, input_args);
#else
    MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
    return mindspore::abstract::kNoShape;
#endif
  }

  TypePtr InferType(const PrimitivePtr &primitive,
                    const std::vector<AbstractBasePtr> & /* input_args */) const override {
    MS_LOG(WARNING) << "This function is the fake infer dtype function and should not be entered. "
                    << "Check the dtype of the output of the operator: "
                    << GetValue<std::string>(primitive->GetAttr("func_name"));

    return TypePtr();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr & /* engine */, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    if (!primitive->isa<PrimitivePy>()) {
      MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
    }
    // cppcheck-suppress unreadVariable
    py::gil_scoped_acquire acquire;
    auto prim_py = dyn_cast<PrimitivePy>(primitive);
    auto py_args = PreparePyInputs(input_args);
    auto output = prim_py->RunInfer(py_args);

    if (!primitive->HasAttr(kCppInferShapeAttr)) {
      return abstract::PyInferRes2Abstract(prim_py, output);
    }

    auto res_dtype = output[kDTypeAttr].cast<TypePtr>();
    if (res_dtype == nullptr) {
      MS_LOG(EXCEPTION)
        << "For custom ops with cpp infer shape functions, we support the case that the output is a tensor."
        << "Thus the inferred dtype should be a type object, but get inferred dtype in: " << output;
    }
    auto shape = InferShape(primitive, input_args);
    auto res = MakeAbstract(shape, res_dtype);
    return res;
  }
};

BaseShapePtr CustomFuncImplInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
#if !defined(_WIN32) && !defined(_WIN64)
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer shape for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer shape for " << op_type;
  if (!primitive->HasAttr(kCppInferShapeAttr)) {
    if (!primitive->isa<PrimitivePy>()) {
      MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
    }
    const auto &inputs = GetInputAbstract(op_type, input_args);
    // cppcheck-suppress unreadVariable
    py::gil_scoped_acquire acquire;
    auto prim_py = dyn_cast<PrimitivePy>(primitive);
    auto py_args = PreparePyInputs(inputs);
    auto out_dict = prim_py->RunInfer(py_args);
    auto shape_obj = out_dict[ATTR_SHAPE];
    auto res_vec = shape_obj.cast<ShapeVector>();
    return std::make_shared<abstract::Shape>(res_vec);
  }
  auto inputs = GetInputAbstract(op_type, input_args);
  return CustomInferShapeByCPP(primitive, inputs);
#else
  MS_LOG(EXCEPTION) << "Custom Operators of type AOT doesn't support Windows currently";
  return mindspore::abstract::kNoShape;
#endif
}

TypePtr CustomFuncImplInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
#if !defined(_WIN32) && !defined(_WIN64)
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_type = GetValue<std::string>(primitive->GetAttr(kRegOpName));
  MS_LOG(DEBUG) << "Start infer type for " << op_type;
  MS_VLOG(VL_CUSTOM_OP) << "Start infer type for " << op_type;
  if (!primitive->isa<PrimitivePy>()) {
    MS_LOG(EXCEPTION) << "The prim is not a PrimitivePy. Prim name: " << primitive->name();
  }
  const auto &inputs = GetInputAbstract(op_type, input_args);
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

struct CustomRegFunc {
  CustomRegFunc() {
    CustomExtFuncImpl::set_infer_shape_call_func(CustomFuncImplInferShape);
    CustomExtFuncImpl::set_infer_type_call_func(CustomFuncImplInferType);
  }
} g_reg_func;

REGISTER_PRIMITIVE_OP_CPP_INFER_IMPL(Custom, prim::kPrimCustom, AGCustomInfer, false);
}  // namespace ops
}  // namespace mindspore
