${return_type} ${op_name}CPU::Call(${call_args_with_type}) {
  ${call_impl}
}
MS_REG_PYBOOST_OP(CPU, ${op_name});
${register_custom_kernel}
