#include "plugin/device/ascend/kernel/internal/moe_token_unpermute.h"

#include <memory>
#include "common/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalMoeTokenUnpermute::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                const internal::OutputsImmutableInfoList &outputs_ii,
                                                                const std::vector<KernelTensor *> &ms_inputs,
                                                                const std::vector<KernelTensor *> &ms_outputs) {
  return internal::CreateMoeTokenUnpermuteOp(inputs_ii, outputs_ii, internal::kInternalMoeTokenUnpermuteOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(MoeTokenUnpermute, internal::kInternalMoeTokenUnpermuteOpName,
                               InternalMoeTokenUnpermute);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MoeTokenUnpermute, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MoeTokenUnpermute, OUTPUT_NUM_1, INDEX_0);

}  // namespace kernel
}  // namespace mindspore