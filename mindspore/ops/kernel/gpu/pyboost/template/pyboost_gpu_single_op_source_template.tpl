${return_type} ${op_name}GPU::Call(${call_args_with_type}) {
  ${call_impl}
}
MS_REG_PYBOOST_OP(GPU, ${op_name});
${register_custom_kernel}
