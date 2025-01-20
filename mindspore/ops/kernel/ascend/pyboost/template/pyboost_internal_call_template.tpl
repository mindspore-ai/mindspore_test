if (internal_op_set.count(op_name()) != 0) {
    InternalAscendCall(op, ${internal_call_args});
    op->CreateOutputSimpleInfo();
    return ${return_values};
}