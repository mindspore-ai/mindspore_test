if (acme_op_set.count(op_name()) != 0) {
    auto op = get_op();
    InferOutput(${acme_call_args});
    ${value_tuple_convert}
    ${create_input_address}
    PyBoostUtils::PrepareOpOutputs(device_context_, op->stream_id(), outputs_);
    AcmeAscendCall(op, ${acme_call_args});
    op->CreateOutputSimpleInfo();
    return ${return_values};
}
