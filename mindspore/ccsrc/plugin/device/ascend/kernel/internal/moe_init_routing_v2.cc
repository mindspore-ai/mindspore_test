#include "plugin/device/ascend/kernel/internal/moe_init_routing_v2.h"

#include <memory>
#include "common/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalMoeInitRoutingV2::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                               const internal::OutputsImmutableInfoList &outputs_ii,
                                                               const std::vector<KernelTensor *> &ms_inputs,
                                                               const std::vector<KernelTensor *> &ms_outputs) {
  internal::MoeInitRoutingParam param;
  param.active_num = ms_inputs[kIndex2]->GetValueWithCheck<int64_t>();
  param.expert_capacity = ms_inputs[kIndex3]->GetValueWithCheck<int64_t>();
  param.expert_num = ms_inputs[kIndex4]->GetValueWithCheck<int64_t>();
  param.drop_pad_mode = ms_inputs[kIndex5]->GetValueWithCheck<int64_t>();
  param.expert_tokens_count_or_cumsum_flag = ms_inputs[kIndex6]->GetValueWithCheck<int64_t>();
  param.expert_tokens_before_capacity_flag = ms_inputs[kIndex7]->GetValueWithCheck<bool>();
  return internal::CreateMoeInitRoutingOp(inputs_ii, outputs_ii, param, internal::kInternalMoeInitRoutingOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(MoeInitRoutingV2, internal::kInternalMoeInitRoutingOpName, InternalMoeInitRoutingV2);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MoeInitRoutingV2, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MoeInitRoutingV2, OUTPUT_NUM_4, INDEX_0, INDEX_1, INDEX_2, INDEX_3);

}  // namespace kernel
}  // namespace mindspore