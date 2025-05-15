${return_type} Internal${op_name}Ascend::Call(${call_args_with_type}) {
    Internal${op_name}AscendCustomize(get_op(), ${call_args});
    return ${return_values};
}
MS_REG_PYBOOST_INTERNAL_OP(Ascend, ${op_name});

