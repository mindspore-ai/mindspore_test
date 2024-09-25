if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

file(GLOB_RECURSE MS_OPTIMIZER_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./mindir/*.cc"
        "./ge/*.cc"
        "./heterogeneous/*.cc"
        "./ge_backend_optimization.cc"
        "./ge_optimization.cc"
        "./get_value_helper.cc"
        "./backend_common_unify_mindir.cc"
        "./format_type/insert_cast.cc"
        "./format_type/set_fracz_group_attr.cc"
        "./format_type/utils.cc"
        "./format_type/deal_ref_output.cc"
        "./common/gllo_utils.cc"
        "./ir_fission/cdist_fission.cc"
        "./ir_fission/bn_split.cc"
        "./ir_fission/bn_grad_split.cc"
        "./ir_fission/seed_adapter.cc"
        "./ir_fission/tensor_scatter_fission.cc"
        "./ir_fission/adam_weight_decay_fission.cc"
        "./ir_fission/batch_norm_grad_infer_fission.cc"
        "./ir_fusion/batchnorm_to_bninfer.cc"
        "./ir_fission/ascend_convert_tuple_input_to_dynamic_input.cc"
        "./ir_fusion/batchnormgrad_to_bninfergrad.cc"
        "./ir_fusion/histogram_fixed_width_fusion.cc"
        "./ir_fusion/adaptive_max_pool2d_fusion.cc"
        "./ir_fusion/flash_attention_fusion.cc"
        "./ir_fusion/mc2_fusion.cc"
        "./ir_fusion/inference_weight_preprocess_utils.cc"
        "./ir_fusion_infer/inference_matmul_split_fusion.cc"
        "./ir_fusion_infer/inference_swiglu_fusion.cc"
        "./ir_fusion/inference_qbmm_fusion_base.cc"
        "./ir_fusion_infer/inference_qbmm_add_fusion.cc"
        "./ir_fusion_infer/inference_qbmm_allreduce_add_fusion.cc"
        "./ir_fusion_infer/shape_reshape_fusion.cc"
        "./ir_fusion_infer/add_layer_norm_fusion.cc"
        "./ir_fusion_infer/add_rms_norm_fusion.cc"
        "./ir_fusion_infer/rms_norm_quant_fusion.cc"
        "./ir_fusion_infer/add_rms_norm_quant_fusion.cc"
        "./ir_fusion_infer/add_cast_rms_norm_cast_fusion.cc"
        "./ir_fusion_infer/matmul_allreduce_fusion.cc"
        "./ir_fusion_infer/split_concat_fusion.cc"
        "./ir_fusion_infer/matmul_elemwise_fusion.cc"
        "./enhancer/eliminate_maketuple_getitem.cc"
        )

set_property(SOURCE ${MS_OPTIMIZER_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)

add_library(_mindspore_plugin_device_ascend_optimizer_910b_obj OBJECT ${MS_OPTIMIZER_910B})
