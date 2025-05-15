${return_type} Internal${op_name}Ascend::Call(${call_args_with_type}) {
    auto op = get_op();
    InferOutput(${internal_call_args});
    ${value_tuple_convert}
    ${const_number_convert}
    ${create_input_address}
    ${create_output_address}
    internal_${operator_name}(op, ${internal_real_call_args});
    return ${return_values};
}
MS_REG_PYBOOST_INTERNAL_OP(Ascend, ${op_name});

