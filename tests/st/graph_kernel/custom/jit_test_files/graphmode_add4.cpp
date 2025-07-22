#include <vector>
#include <map>
#include <string>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

#include "kernel/ascend/opapi/aclnn/add_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

#include "ops/ops_func_impl/op_func_impl.h"

#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

#include "ms_extension/api.h"
#include "module.h"

namespace mindspore::ops {
static inline bool IsIntegralBinaryType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

class OPS_API Add4OpFuncImpl : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto output_shape = input_infos[kInputIndex0]->GetShape();
    std::vector<std::string> input_names = {"input", "other", "alpha"};
    //  for (size_t i = 1; i < input_infos.size(); ++i) {
    //    auto input_shape = input_infos[i]->GetShape();
    //    output_shape = CalBroadCastShape(output_shape, input_shape, primitive->name(), input_names[i - 1],
    //    input_names[i]);
    //  }
    return {output_shape};
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override {
    auto input = input_infos[kInputIndex0]->GetType();
    auto other = input_infos[kInputIndex1]->GetType();
    auto alpha = input_infos[kInputIndex2]->GetType();
    auto typePtr = TypeIdToType(input);
    if (alpha == kNumberTypeFloat32 && (IsIntegralBinaryType(input) || IsIntegralBinaryType(other))) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                              << "', floating alpha need floating input and other, but got " << TypeIdToString(input)
                              << " and " << TypeIdToString(other);
    }
    if (alpha == kNumberTypeBool && (input != kNumberTypeBool || other != kNumberTypeBool)) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                              << "', boolean alpha need boolean input and other, but got " << TypeIdToString(input)
                              << " and " << TypeIdToString(other);
    }
    if (input != other) {
      MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                              << "], the input arguments must have same data type, but got Tensor["
                              << TypeIdToString(input) << "] and Tensor[" << TypeIdToString(other) << "]";
    }
    if (!std::any_of(common_valid_types_with_complex_and_bool.begin(), common_valid_types_with_complex_and_bool.end(),
                     [&typePtr](TypePtr accept) { return typePtr == accept; })) {
      MS_EXCEPTION(TypeError) << "For primitive[" << primitive->name()
                              << "], the input arguments must have error data type, got Tensor["
                              << TypeIdToString(input) << "] and Tensor[" << TypeIdToString(other) << "]";
    }
    return {input};
  }

  bool GeneralInferRegistered() const override { return true; }
};

}  // namespace mindspore::ops

namespace mindspore {
namespace kernel {

class Add4Ascend : public AclnnKernelMod {
 public:
  Add4Ascend() : AclnnKernelMod(std::move("aclnnAdd")) {}

  ~Add4Ascend() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], alpha_, outputs[kIndex0]);
    return true;
  }

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    auto alpha_dtype_id = inputs[kIndex2]->dtype_id();
    switch (alpha_dtype_id) {
      case kNumberTypeBool: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<bool>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeFloat32: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<float>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeFloat64: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<double>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      case kNumberTypeInt64: {
        auto alpha_value = inputs[kIndex2]->GetValueWithCheck<int64_t>();
        MAKE_SCALAR(alpha_value, alpha_dtype_id, alpha_);
        break;
      }
      default:
        MS_LOG(EXCEPTION) << "AddExt only support bool, float32, float64 and int64, but got " << alpha_dtype_id;
    }
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], alpha_, outputs[kIndex0]);
  }

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  ScalarPtr alpha_ = nullptr;
};
}  // namespace kernel
}  // namespace mindspore

MS_CUSTOM_OP_REGISTER(add4, Add4OpFuncImpl, Add4Ascend)

REG_GRAPH_MODE_OP(add4);
