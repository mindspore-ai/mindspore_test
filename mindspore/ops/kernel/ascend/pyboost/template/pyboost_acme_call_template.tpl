if (acme_op_set.count(op_name()) != 0) {
    AcmeAscendCall(op, ${acme_call_args});
    op->CreateOutputSimpleInfo();
    return ${return_values};
}