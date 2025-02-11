/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "plugin/device/ascend/optimizer/ir_fission/cdist_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/tensor_scatter_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/adam_weight_decay_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/batch_norm_grad_infer_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/bn_grad_split.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchnorm_to_bninfer.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchnormgrad_to_bninfergrad.h"
#include "plugin/device/ascend/optimizer/mindir/add_depend_for_adamw.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_mindir_op_adapter.h"
#include "plugin/device/ascend/optimizer/mindir/renorm_split.h"
#include "plugin/device/ascend/optimizer/mindir/optimizer_unify_output.h"
#include "plugin/device/ascend/optimizer/mindir/space_batch_nd_attr_update.h"
#include "plugin/device/ascend/optimizer/mindir/bn_grad_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/all_to_all_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/all_to_all_v_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/neighbor_exchange_v2_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/reduce_axis_update.h"
#include "plugin/device/ascend/optimizer/mindir/clip_by_norm_fission.h"
#include "plugin/device/ascend/optimizer/mindir/dropout_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "plugin/device/ascend/optimizer/mindir/adam_weight_decay_unify_mindir.h"
#include "plugin/device/ascend/optimizer/ge/lamb_fission.h"
#include "plugin/device/ascend/optimizer/ge/adjust_print_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/add_attr_to_dump.h"
#include "plugin/device/ascend/optimizer/ge/getnext_for_ge.h"
#include "plugin/device/ascend/optimizer/ir_fusion/adaptive_max_pool2d_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/flash_attention_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_layer_norm_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_rms_norm_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/rms_norm_quant_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_rms_norm_quant_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_cast_rms_norm_cast_quant_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_cast_rms_norm_cast_fusion.h"
#include "plugin/device/ascend/optimizer/ge/avg_pool_grad_for_ge.h"
#include "plugin/device/ascend/optimizer/ir_fusion/mc2_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/insert_depend_for_all_gather.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/shape_reshape_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/split_concat_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/matmul_allreduce_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_matmul_split_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_swiglu_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_qbmm_add_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_qbmm_allreduce_add_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/matmul_allreduce_add_rmsnorm_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/qbmm_allreduce_convert_bias.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/matmul_elemwise_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/remove_fa_tensor_to_tuple_ops.h"
#include "utils/phase.h"
#include "backend/common/graph_kernel/core/graph_kernel_pass_manager.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "plugin/device/ascend/optimizer/ir_fusion/batchmatmul_reducescatter_alltoall_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fusion/alltoall_allgather_batch_matmul_fusion.h"

namespace mindspore {
namespace opt {
void GetBackendCommonUnifyMindIRPassManager(PassManagerPtr *unify_mindir_pm) {
  MS_EXCEPTION_IF_NULL(unify_mindir_pm);
  (*unify_mindir_pm)->AddPass(std::make_shared<RenormSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::ReduceAxisUpdate>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::ClipByNormFission>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SpaceToBatchNDAttrUpdate>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::BatchToSpaceNDAttrUpdate>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AdamWeightDecayUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddDependForAdamW>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<CdistFission>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<CdistGradFission>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::BatchMatMulReduceScatterAllToAllFusion>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::AllToAllAllGatherBatchMatMulFusion>());

  // Since the SparseSoftmaxCrossEntropyWithLogits operator can only use AICPU and has poor execution performance,
  // it does not take effect for the time being.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool graph_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
  if (graph_mode) {
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::GradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  } else {
    // Add PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR pass first to avoid the backward loss function
    // from the python frontend matching the pattern defined in
    // PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR.
    // TODO(hbhu_bin): In mindspore, SparseSoftmaxCrossEntropyWithLogits has different outputs based on the "is_grad"
    // attribute, but it has two outputs in CANN. These pass cann be removed when convert "is_grad" attribute to input.
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIRV2>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeGradSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
    (*unify_mindir_pm)->AddPass(std::make_shared<opt::PynativeSparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>());
  }

  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutExtUnifyMindIR1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutGradExtUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutUnifyMindIR1>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::DropoutGradUnifyMindIR>());
  // AllToAll & AlltoAllV
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeV2UnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::NeighborExchangeV2GradUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AllToAllUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AlltoAllVUnifyMindIR>());
  // batchnorm
  (*unify_mindir_pm)->AddPass(std::make_shared<BnSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::BatchNormGradUnifyMindIR>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BnGradSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BatchNormGrad2BNInferGrad>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<BatchNormGradInferFission>());
  (*unify_mindir_pm)->AddPass(std::make_shared<BatchNorm2BNInfer>());

  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::LambFissionGe>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AdjustPrintForGe>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::GetNextForGE>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SyncBnSplit>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::SyncBnGradSplit>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::AdaptiveMaxPool2DGeFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AvgPoolGradForGE>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::MatmulReduceScatterFusion>());
  (*unify_mindir_pm)->AddFusionPass(std::make_shared<opt::AllGatherMatmulFusion>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AddAttrToDump>());
  (*unify_mindir_pm)->AddPass(std::make_shared<opt::AscendMindIROpAdapter>());
}

PassManagerPtr GetBackendFusionGroupPassManager() {
  auto pm = std::make_shared<PassManager>("unify_mindir_2");
  pm->AddFusionPass(std::make_shared<opt::FlashAttentionFusionV1>());
  pm->AddFusionPass(std::make_shared<opt::FlashAttentionFusionV2>());
  pm->AddFusionPass(std::make_shared<opt::QuantBatchMatmulAllReduceFusion>());
  pm->AddFusionPass(std::make_shared<opt::MatMulAllReduceFusion>());
  pm->AddFusionPass(std::make_shared<opt::MatMulAllReduceAddRmsNormFusion>());

#ifdef ENABLE_INTERNAL_KERNELS
  pm->AddFusionPass(std::make_shared<opt::AddLayernormFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddLayernormV3Fusion>());
  pm->AddFusionPass(std::make_shared<opt::AddLayernormExtFusion>());
  pm->AddFusionPass(std::make_shared<opt::QbmmAddFusion>());
  pm->AddFusionPass(std::make_shared<opt::InferenceSwiGLUFusion>());
  pm->AddFusionPass(std::make_shared<opt::InferenceMatmulSplitFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddRmsNormDynamicQuantFusion>());
  pm->AddFusionPass(std::make_shared<opt::ShapeReshapeFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddRmsNormQuantFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddCastRmsNormCastQuantFusion>());
  pm->AddFusionPass(std::make_shared<opt::RmsNormQuantFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddRmsNormFusion>());
  pm->AddFusionPass(std::make_shared<opt::AddCastRmsNormCastFusion>());
  pm->AddFusionPass(std::make_shared<opt::SplitConcatFusion>());
  pm->AddFusionPass(std::make_shared<opt::MatmulElemFusion>());
  pm->AddFusionPass(std::make_shared<opt::QbmmAllReduceConvertBias>());
  pm->AddFusionPass(std::make_shared<opt::QbmmAllReduceAddFusion>());
  pm->AddFusionPass(std::make_shared<opt::RemoveFATensorToTupleOps>());
#endif  // ENABLE_INTERNAL_KERNELS
  return pm;
}

void AscendUnfoldInputsForSpecialNodes(const KernelGraphPtr &kernel_graph) {
  uint64_t start_time = profiler::GetClockSyscnt();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_before_unfold_inputs_for_special_nodes_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
    DumpIRProto(kernel_graph,
                "before_unfold_inputs_for_special_nodes_hwopt_" + std::to_string(kernel_graph->graph_id()));
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto unfold_inputs_pm = std::make_shared<opt::PassManager>("unfold_inputs_for_special_nodes_pm");
  unfold_inputs_pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());

  optimizer->AddPassManager(unfold_inputs_pm);
  (void)optimizer->Optimize(kernel_graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name =
      "hwopt_d_after_unfold_inputs_for_special_nodes_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph, true, kWholeStack);
  }
#endif
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UnfoldInputsForSpecialNodes",
                                  start_time, profiler::GetClockSyscnt(), 0);
}
}  // namespace opt
}  // namespace mindspore
