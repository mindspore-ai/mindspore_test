/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/eltwise_grad_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <string>
#include <functional>
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/lite_ops.h"
#include "mindspore/ops/op_def/arithmetic_ops.h"
#include "include/common/thread_pool.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "nnacl/fp32_grad/activation_grad_fp32.h"
#include "nnacl/fp32_grad/arithmetic_grad.h"
#include "nnacl/errorcode.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace kernel {
namespace eltwise_grad_cpu {
namespace {
constexpr auto kReluGrad = "ReluGrad";
constexpr auto kReLU6Grad = "ReLU6Grad";
constexpr auto kAbsGrad = "AbsGrad";
constexpr auto kSigmoidGrad = "SigmoidGrad";
constexpr auto kSqrtGrad = "SqrtGrad";
constexpr auto kTanhGrad = "TanhGrad";
constexpr auto kGeLUGrad = "GeLUGrad";
constexpr auto kAsinGrad = "AsinGrad";
constexpr auto kACosGrad = "ACosGrad";
constexpr auto kAtanGrad = "AtanGrad";
constexpr auto kAsinhGrad = "AsinhGrad";
constexpr auto kInvGrad = "InvGrad";
constexpr auto kAcoshGrad = "AcoshGrad";
constexpr auto kSoftplusGrad = "SoftplusGrad";
constexpr auto kRsqrtGrad = "RsqrtGrad";
constexpr auto kReciprocalGrad = "ReciprocalGrad";

template <typename T>
class EltWiseGradCpuTypeFunc : public CpuKernelFunc {
 public:
  EltWiseGradCpuTypeFunc() = default;
  ~EltWiseGradCpuTypeFunc() override = default;
  void InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                const std::vector<KernelTensor *> &outputs) override;
  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
               const std::vector<KernelTensor *> &outputs) override;

 private:
  void ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexSigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void RsqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ReciprocalGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexAtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexAsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void InvGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexAsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void ComplexAcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;
  void SoftplusGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const;

  using TypeComputeFunc = std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>;
  TypeComputeFunc compute_func_{nullptr};
  std::string kernel_name_;
};

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ReluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (std::is_same<T, float>::value) {
    int ret = ::ReluGrad(input1 + start, input2 + start, end - start, out + start);
    if (ret == NNACL_ERR) {
      MS_LOG(EXCEPTION) << "For 'ReLUGrad', execute failed. Error no: " << ret;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      out[i] = (input2[i] > T(0)) ? input1[i] : static_cast<T>(0);
    }
  }
}

template <>
void EltWiseGradCpuTypeFunc<bool>::ReluGrad(const bool *input1, const bool *input2, bool *out, size_t start,
                                            size_t end) const {
  for (size_t i = start; i < end; i++) {
    out[i] = input2[i] ? input1[i] : false;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ReLU6Grad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'ReLU6Grad', the dtype of input must be float.";
  }

  int ret = ::Relu6Grad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "For 'ReLU6Grad', execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::AbsGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (std::is_same<T, float>::value) {
    int ret = ::ElementAbsGrad(input1 + start, input2 + start, out + start, end - start);
    if (ret == NNACL_ERR) {
      MS_LOG(EXCEPTION) << "For 'AbsGrad', execute failed. Error no: " << ret;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      // cppcheck-suppress unsignedLessThanZero
      out[i] = (input1[i] < 0) ? -input2[i] : ((input1[i] > 0) ? input2[i] : 0);
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::SigmoidGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (std::is_same<T, float>::value) {
    int ret = ::SigmoidGrad(input2 + start, input1 + start, end - start, out + start);
    if (ret == NNACL_ERR) {
      MS_LOG(EXCEPTION) << "For 'SigmoidGrad', execute failed. Error no: " << ret;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      T dividend = input2[i];
      T divisor = input1[i] * (static_cast<T>(1) - input1[i]);
      out[i] = dividend * divisor;
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexSigmoidGrad(const T *input1, const T *input2, T *out, size_t start,
                                                   size_t end) const {
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    for (size_t i = start; i < end; i++) {
      T dividend = input2[i];
      T divisor = std::conj(input1[i] * (static_cast<T>(1) - input1[i]));
      out[i] = dividend * divisor;
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::SqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    for (size_t i = start; i < end; i++) {
      constexpr T coff = static_cast<T>(2);
      out[i] = (input2[i] / std::conj(input1[i] * coff));
    }
  } else {
    for (size_t i = start; i < end; i++) {
      constexpr T coff = static_cast<T>(2);
      out[i] = input2[i] / (input1[i] * coff);
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::RsqrtGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    for (size_t i = start; i < end; i++) {
      constexpr T coff = static_cast<T>(-2);
      out[i] = (conj(input1[i]) * conj(input1[i]) * conj(input1[i])) * (input2[i] / coff);
    }
  } else {
    for (size_t i = start; i < end; i++) {
      T coff = static_cast<T>(-2);
      out[i] = (input1[i] * input1[i] * input1[i]) * (input2[i] / coff);
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ReciprocalGrad(const T *input1, const T *input2, T *out, size_t start,
                                               size_t end) const {
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(-1) * conj(input1[i] * input1[i]) * input2[i];
    }
  } else {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(-1) * input1[i] * input1[i] * input2[i];
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::TanhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (std::is_same<T, float>::value) {
    int ret = ::TanhGrad(input2 + start, input1 + start, end - start, out + start);
    if (ret == NNACL_ERR) {
      MS_LOG(EXCEPTION) << "For 'TanhGrad', execute failed. Error no: " << ret;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      T dividend = input2[i];
      T divisor = static_cast<T>(1) - input1[i] * input1[i];
      out[i] = dividend * divisor;
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::GeluGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T x = input2[i];
    auto double_x = static_cast<T>(x);
    T tanh_res = static_cast<T>(std::tanh(0.7978845608 * (double_x + 0.044715 * double_x * double_x * double_x)));
    T mul_right = static_cast<T>(0.7978845608 + 0.1070322244 * double_x * double_x);
    T y_res = ((static_cast<T>(1.0) + tanh_res) + x * (static_cast<T>(1.0) - tanh_res * tanh_res) * mul_right) /
              static_cast<T>(2.0);
    out[i] = input1[i] * y_res;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexAsinGrad(const T *input1, const T *input2, T *out, size_t start,
                                                size_t end) const {
  if constexpr (std::is_same_v<T, complex64>) {
    for (size_t i = start; i < end; i++) {
      T dividend = input2[i];
      T divisor = std::conj(sqrt(static_cast<T>(1) - input1[i] * input1[i]));
      if (divisor == static_cast<T>(0)) {
        float a = std::numeric_limits<float>::quiet_NaN();
        std::complex<float> nancp(a, a);
        out[i] = nancp;
        continue;
      }
      out[i] = dividend / divisor;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      T dividend = input2[i];
      T divisor = std::conj(sqrt(static_cast<T>(1) - input1[i] * input1[i]));
      if (divisor == static_cast<T>(0)) {
        double a = std::numeric_limits<double>::quiet_NaN();
        std::complex<double> nancp(a, a);
        out[i] = nancp;
        continue;
      }
      out[i] = dividend / divisor;
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::AsinGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sqrt(1 - input1[i] * input1[i]);
    if (std::equal_to<T>()(divisor, 0)) {
      if (std::equal_to<T>()(dividend, 0)) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ACosGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = -input2[i];
    T divisor = sqrt(1 - input1[i] * input1[i]);
    if (std::equal_to<T>()(divisor, 0)) {
      if (std::equal_to<T>()(dividend, 0)) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexACosGrad(const T *input1, const T *input2, T *out, size_t start,
                                                size_t end) const {
  if constexpr (std::is_same_v<T, complex64>) {
    for (size_t i = start; i < end; i++) {
      T dividend = -input2[i];
      T divisor = std::conj(sqrt(static_cast<T>(1) - input1[i] * input1[i]));
      if (divisor == static_cast<T>(0)) {
        float a = std::numeric_limits<float>::quiet_NaN();
        std::complex<float> nancp(a, a);
        out[i] = nancp;
        continue;
      }
      out[i] = dividend / divisor;
    }
  } else {
    for (size_t i = start; i < end; i++) {
      T dividend = -input2[i];
      T divisor = std::conj(sqrt(static_cast<T>(1) - input1[i] * input1[i]));
      if (divisor == static_cast<T>(0)) {
        double a = std::numeric_limits<double>::quiet_NaN();
        std::complex<double> nancp(a, a);
        out[i] = nancp;
        continue;
      }
      out[i] = dividend / divisor;
    }
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::AtanGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  auto one = static_cast<T>(1);
  auto zero = static_cast<T>(0);
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = one + input1[i] * input1[i];
    if (std::equal_to<T>()(divisor, zero)) {
      if (std::equal_to<T>()(dividend, zero)) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexAtanGrad(const T *input1, const T *input2, T *out, size_t start,
                                                size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = static_cast<T>(1) + input1[i] * input1[i];
    if (std::equal_to<T>()(divisor, static_cast<T>(0))) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    out[i] = dividend / conj(divisor);
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::AsinhGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = cosh(input1[i]);
    if (std::equal_to<T>()(divisor, 0)) {
      if (std::equal_to<T>()(dividend, 0)) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexAsinhGrad(const T *input1, const T *input2, T *out, size_t start,
                                                 size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = std::conj(cosh(input1[i]));
    if (std::equal_to<T>()(divisor, 0)) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::InvGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T neg_one = static_cast<T>(-1);
    out[i] = neg_one * input2[i] * input1[i] * input1[i];
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::AcoshGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = sinh(input1[i]);
    if (std::equal_to<T>()(divisor, 0)) {
      if (std::equal_to<T>()(dividend, 0)) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > 0 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::ComplexAcoshGrad(const T *input1, const T *input2, T *out, size_t start,
                                                 size_t end) const {
  for (size_t i = start; i < end; i++) {
    T dividend = input2[i];
    T divisor = std::conj(sinh(input1[i]));
    if (std::equal_to<T>()(divisor, 0)) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::SoftplusGrad(const T *input1, const T *input2, T *out, size_t start, size_t end) const {
  if constexpr (!std::is_same<T, float>::value) {
    MS_LOG(EXCEPTION) << "For 'SoftplusGrad', the dtype of input must be float.";
  }

  int ret = ::SoftplusGrad(input1 + start, input2 + start, end - start, out + start);
  if (ret == NNACL_ERR) {
    MS_LOG(EXCEPTION) << "For 'SoftplusGrad', execute failed. Error no: " << ret;
  }
}

template <typename T>
void EltWiseGradCpuTypeFunc<T>::InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &) {
  kernel_name_ = primitive->name();
  if constexpr (std::is_same_v<T, double>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimSqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::SqrtGrad},
              {prim::kPrimGeLUGrad->name(), &EltWiseGradCpuTypeFunc<T>::GeluGrad},
              {prim::kPrimAsinGrad->name(), &EltWiseGradCpuTypeFunc<T>::AsinGrad},
              {prim::kPrimACosGrad->name(), &EltWiseGradCpuTypeFunc<T>::ACosGrad},
              {prim::kPrimRsqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::RsqrtGrad},
              {prim::kPrimReciprocalGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReciprocalGrad},
              {prim::kPrimAtanGrad->name(), &EltWiseGradCpuTypeFunc<T>::AtanGrad},
              {prim::kPrimTanhGrad->name(), &EltWiseGradCpuTypeFunc<T>::TanhGrad},
              {prim::kPrimAsinhGrad->name(), &EltWiseGradCpuTypeFunc<T>::AsinhGrad},
              {prim::kPrimInvGrad->name(), &EltWiseGradCpuTypeFunc<T>::InvGrad},
              {prim::kPrimAcoshGrad->name(), &EltWiseGradCpuTypeFunc<T>::AcoshGrad},
              {prim::kPrimAbsGrad->name(), &EltWiseGradCpuTypeFunc<T>::AbsGrad},
              {prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad},
              {prim::kPrimSigmoidGrad->name(), &EltWiseGradCpuTypeFunc<T>::SigmoidGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "For 'EltWiseGrad', it does not support " << kernel_name_ << " with double as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr (std::is_same_v<T, float16>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimAtanGrad->name(), &EltWiseGradCpuTypeFunc<T>::AtanGrad},
              {prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad},
              {prim::kPrimReciprocalGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReciprocalGrad},
              {prim::kPrimRsqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::RsqrtGrad},
              {prim::kPrimSigmoidGrad->name(), &EltWiseGradCpuTypeFunc<T>::SigmoidGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGradCpu does not support " << kernel_name_ << " with float as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr (std::is_same_v<T, float>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimReLU6Grad->name(), &EltWiseGradCpuTypeFunc<T>::ReLU6Grad},
              {prim::kPrimSigmoidGrad->name(), &EltWiseGradCpuTypeFunc<T>::SigmoidGrad},
              {prim::kPrimAbsGrad->name(), &EltWiseGradCpuTypeFunc<T>::AbsGrad},
              {prim::kPrimTanhGrad->name(), &EltWiseGradCpuTypeFunc<T>::TanhGrad},
              {prim::kPrimSqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::SqrtGrad},
              {prim::kPrimGeLUGrad->name(), &EltWiseGradCpuTypeFunc<T>::GeluGrad},
              {prim::kPrimAsinGrad->name(), &EltWiseGradCpuTypeFunc<T>::AsinGrad},
              {prim::kPrimACosGrad->name(), &EltWiseGradCpuTypeFunc<T>::ACosGrad},
              {prim::kPrimAtanGrad->name(), &EltWiseGradCpuTypeFunc<T>::AtanGrad},
              {prim::kPrimAsinhGrad->name(), &EltWiseGradCpuTypeFunc<T>::AsinhGrad},
              {prim::kPrimInvGrad->name(), &EltWiseGradCpuTypeFunc<T>::InvGrad},
              {prim::kPrimRsqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::RsqrtGrad},
              {prim::kPrimReciprocalGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReciprocalGrad},
              {prim::kPrimAcoshGrad->name(), &EltWiseGradCpuTypeFunc<T>::AcoshGrad},
              {prim::kPrimSoftplusGrad->name(), &EltWiseGradCpuTypeFunc<T>::SoftplusGrad},
              {prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "For 'EltWiseGrad', it does not support " << kernel_name_ << " with float as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr ((std::is_same_v<T, int>) || (std::is_same_v<T, int8_t>) || (std::is_same_v<T, int16_t>) ||
                (std::is_same_v<T, int64_t>)) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimAbsGrad->name(), &EltWiseGradCpuTypeFunc<T>::AbsGrad},
              {prim::kPrimInvGrad->name(), &EltWiseGradCpuTypeFunc<T>::InvGrad},
              {prim::kPrimRsqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::RsqrtGrad},
              {prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "For 'EltWiseGrad', it does not support " << kernel_name_ << " with int as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
  }
  if constexpr ((std::is_same_v<T, uint8_t>) || (std::is_same_v<T, uint16_t>) || (std::is_same_v<T, uint32_t>) ||
                (std::is_same_v<T, uint64_t>)) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGrad Cpu does not support " << kernel_name_ << " with uint as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr (std::is_same_v<T, bool>) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimReluGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReluGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "EltWiseGrad Cpu does not support " << kernel_name_ << " with bool as input.";
    }
    compute_func_ = elt_map.at(kernel_name_);
    return;
  }
  if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
    static const std::map<std::string,
                          std::function<void(EltWiseGradCpuTypeFunc *, const T *, const T *, T *, size_t, size_t)>>
      elt_map{{prim::kPrimAcoshGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexAcoshGrad},
              {prim::kPrimAsinhGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexAsinhGrad},
              {prim::kPrimAsinGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexAsinGrad},
              {prim::kPrimACosGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexACosGrad},
              {prim::kPrimTanhGrad->name(), &EltWiseGradCpuTypeFunc<T>::TanhGrad},
              {prim::kPrimAtanGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexAtanGrad},
              {prim::kPrimInvGrad->name(), &EltWiseGradCpuTypeFunc<T>::InvGrad},
              {prim::kPrimSqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::SqrtGrad},
              {prim::kPrimReciprocalGrad->name(), &EltWiseGradCpuTypeFunc<T>::ReciprocalGrad},
              {prim::kPrimRsqrtGrad->name(), &EltWiseGradCpuTypeFunc<T>::RsqrtGrad},
              {prim::kPrimSigmoidGrad->name(), &EltWiseGradCpuTypeFunc<T>::ComplexSigmoidGrad}};
    if (elt_map.find(kernel_name_) == elt_map.end()) {
      MS_LOG(EXCEPTION) << "For 'EltWiseGrad', it does not support " << kernel_name_;
    }
    compute_func_ = elt_map.at(kernel_name_);
  }
}

template <typename T>
bool EltWiseGradCpuTypeFunc<T>::RunFunc(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  const auto input0 = reinterpret_cast<T *>(inputs[0]->device_ptr());
  const auto input1 = reinterpret_cast<T *>(inputs[1]->device_ptr());
  auto output = reinterpret_cast<T *>(outputs[0]->device_ptr());

  ParallelLaunchAutoSearch(
    std::bind(compute_func_, this, input0, input1, output, std::placeholders::_1, std::placeholders::_2),
    outputs[0]->size() / sizeof(T), this, &parallel_search_info_);
  return true;
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeEltWiseGradFunc() {
  return std::make_shared<EltWiseGradCpuTypeFunc<T>>();
}

using FuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, FuncCreator>>> kernel_attr_list_map = {
  {kReluGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &SpecializeEltWiseGradFunc<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SpecializeEltWiseGradFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &SpecializeEltWiseGradFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SpecializeEltWiseGradFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &SpecializeEltWiseGradFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &SpecializeEltWiseGradFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &SpecializeEltWiseGradFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &SpecializeEltWiseGradFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &SpecializeEltWiseGradFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpecializeEltWiseGradFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>}}},
  {kReLU6Grad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>}}},
  {kAbsGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SpecializeEltWiseGradFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SpecializeEltWiseGradFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>}}},
  {kSigmoidGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpecializeEltWiseGradFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kSqrtGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kTanhGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kGeLUGrad,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>}}},
  {kGeLUGrad,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>}}},
  {kAsinGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kACosGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kAtanGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpecializeEltWiseGradFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kAsinhGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kInvGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SpecializeEltWiseGradFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SpecializeEltWiseGradFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &SpecializeEltWiseGradFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kAcoshGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}},
  {kSoftplusGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>}}},
  {kRsqrtGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &SpecializeEltWiseGradFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &SpecializeEltWiseGradFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpecializeEltWiseGradFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>}}},
  {kReciprocalGrad,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &SpecializeEltWiseGradFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SpecializeEltWiseGradFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SpecializeEltWiseGradFunc<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SpecializeEltWiseGradFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SpecializeEltWiseGradFunc<complex128>}}}};
}  // namespace

bool EltWiseGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto iter = kernel_attr_list_map.find(kernel_name_);
  if (iter == kernel_attr_list_map.end()) {
    MS_LOG(ERROR) << "For 'EltWiseGrad', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, FuncCreator>>>(kernel_attr_list_map)
                  << ", but got " << kernel_name_;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'EltWiseGrad', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  func_obj_ = kernel_attr_list_map[kernel_name_][index].second();
  func_obj_->InitFunc(primitive_, inputs, outputs);
  return true;
}

int EltWiseGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

std::vector<KernelAttr> EltWiseGradCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list_map.find(kernel_name_);
  if (iter == kernel_attr_list_map.end()) {
    MS_LOG(ERROR) << "For 'EltWiseGrad', it does not support " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FuncCreator> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReluGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kReluGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU6Grad, []() {
  return std::make_shared<EltWiseGradCpuKernelMod>(prim::kPrimReLU6Grad->name());
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AbsGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kAbsGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SigmoidGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kSigmoidGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SqrtGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kSqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TanhGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kTanhGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, GeLUGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kGeLUGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AsinGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kAsinGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ACosGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kACosGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AtanGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kAtanGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AsinhGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kAsinhGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, InvGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kInvGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AcoshGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kAcoshGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SoftplusGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kSoftplusGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, RsqrtGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kRsqrtGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReciprocalGrad,
                                 []() { return std::make_shared<EltWiseGradCpuKernelMod>(kReciprocalGrad); });
}  // namespace eltwise_grad_cpu
}  // namespace kernel
}  // namespace mindspore
