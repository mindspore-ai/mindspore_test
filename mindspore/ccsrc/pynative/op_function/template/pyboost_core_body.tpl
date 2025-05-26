py::object PYNATIVE_EXPORT ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  op_run_info->source_type = source_type;
  auto py_output = tensor::MakeTuple<tensor::TensorWrapper, ${output_num}>();
  auto promises = tensor::TransformPromise(py_output);

  DispatchOp(
    std::make_shared<PyboostPromiseTask>(
      [${op_args}, prim, promises](const PyboostOpRunInfoPtr &op_run_info) {

        auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
        kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->stream_id);

        // stub tensor to tensor.
        ${convert_stub}
        ${implicit_cast}
        kernel::pyboost::OpRunStatus::Get().set_run_info(
            kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                      op_run_info->async_status.is_jit_compiling,
                                      op_run_info->async_status.custom_bprop_cell_count,
                                      op_run_info->device_target));
        kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);

        auto outputs = kernel::pyboost::${operator_name}(${cast_args});
        kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);

        tensor::SetPromise(promises, outputs);
      }, [promises]() {
        tensor::SetException(promises);
      }, op_run_info));

    MS_LOG(DEBUG) << "Run ${func_name} end";
    return py::reinterpret_steal<py::object>(tensor::TransformOutput(py_output));
}
