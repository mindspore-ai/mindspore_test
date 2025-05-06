#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_MOE_TOKEN_UNPERMUTE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_MOE_TOKEN_UNPERMUTE_H_

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "include/internal.h"

namespace mindspore {
namespace kernel {
class InternalMoeTokenUnpermute : public InternalKernelMod {
 public:
  InternalMoeTokenUnpermute() : InternalKernelMod() {}
  ~InternalMoeTokenUnpermute() = default;

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_MOE_TOKEN_UNPERMUTE_H_