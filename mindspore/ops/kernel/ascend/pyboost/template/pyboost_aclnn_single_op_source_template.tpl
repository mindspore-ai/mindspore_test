${return_type} ${op_name}Ascend::Call(${call_args_with_type}) {
  ${call_impl}
}
MS_REG_PYBOOST_OP(Ascend, ${op_name});
${register_custom_kernel}
