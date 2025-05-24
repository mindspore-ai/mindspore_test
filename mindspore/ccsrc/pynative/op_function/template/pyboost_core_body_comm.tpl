py::object ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
    MS_LOG(DEBUG) << "Run ${func_name} start";
    auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
    std::string target;
    const auto &group_str = GetValue<std::string>(group);
    if (group_str.compare(0, 4, "mccl") == 0) {
        target = "CPU";
    } else {
        target = op_run_info->device_target;
    }
    const auto type = GetPredictOutTypeByName(prim->name());
    static auto top_type = GetPredictOutTypeByOutputNum(op_run_info->op_prim, type);

    auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}>();
    auto promises = tensor::TransformPromise(py_output);

    auto comm_handle_py = std::make_shared<hal::CommHandlePy>(runtime::OpRunner::GetDeviceContext(target));
    auto comm_handle_py_obj = py::cast(comm_handle_py);
    kernel::pyboost::CommHandlePtr comm_handle{nullptr};

    comm_handle_py->comm_handle()->CreateEvent();
    comm_handle = comm_handle_py->comm_handle();
    {
        GilReleaseWithCheck release_gil;
        op_run_info->source_type = source_type;
        DispatchOp(
            std::make_shared<PyboostPromiseTask>(
            [${op_args}, comm_handle, target, promises](const PyboostOpRunInfoPtr &op_run_info) {
                MS_LOG(DEBUG) << "Run frontend task ${func_name} start";
                auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
                kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->stream_id);

                // stub tensor to tensor.
                ${convert_stub}

                // Do mixed precision and implicit cast
                ${implicit_cast}

                // Run op
                auto outputs = kernel::pyboost::${operator_name}_inner(${cast_args}, comm_handle, target);
                auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();
                const auto &op_prim = op->primitive();
                ${optional_to_value}

                // Create output value
                AutoGradUtil::SetInferOutputToGrad(op_run_info, op);
                // Create output value
                auto real_output = AutoGradUtil::Make${is_multi}Output(op_run_info->requires_grad, op${view_arg});
                // Do auto grad
                if (op_run_info->requires_grad) {
                  // Refresh op prim, otherwish the size of inputs will be incorrect.
                  auto op_grad_info = std::make_shared<OpGradInfo>(op_prim, std::vector<ValuePtr>({${grad_args}}), real_output);
                  op_grad_info->output_value_simple_info = op_run_info->output_value_simple_info;
                  PyNativeAlgo::PyBoost::DoGrad(op, op_grad_info, op_run_info->async_status);
                }
                // Data sync in mix mode(Graph and PyNative)
                PyNativeAlgo::PyBoost::DataSyncForGraph(op);
                kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);
                tensor::SetPromise(promises, real_output);
                MS_LOG(DEBUG) << "Run frontend task ${func_name} end";
            },
            [promises]() {
              tensor::SetException(promises);
            }, op_run_info)
        );
        MS_LOG(DEBUG) << "Run ${func_name} end";
    }
    return py::reinterpret_steal<py::object>(tensor::TransformOutput(py_output, comm_handle_py_obj.release().ptr()));
}
