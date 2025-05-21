py::object PYNATIVE_EXPORT ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init_Pyboost(prim);
  op_run_info->source_type = source_type;

  // TODO: Not support multi-thread yet.
  {
  GilReleaseWithCheck no_gil;
  runtime::Pipeline::Get().frontend_stage()->Wait();
  }

  auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
  kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->stream_id);

  // stub tensor to tensor.
  ${convert_stub}
  
  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                op_run_info->async_status.is_jit_compiling,
                                op_run_info->async_status.custom_bprop_cell_count,
                                op_run_info->device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);

  auto outputs = [&](){
    // Do mixed precision and implicit cast
    static const std::vector<std::vector<size_t>> same_type_table{${same_type}};
    auto [${cast_args}] = PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<${type_num}>(op_run_info, prim->name(),
                                                                                      same_type_table, ${call_args});

    GilReleaseWithCheck no_gil;
    return kernel::pyboost::${operator_name}(${cast_args});
  }();
  auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();
  // Data sync in mix mode(Graph and PyNative)
  PyNativeAlgo::PyBoost::DataSyncForGraph(op);
  kernel::pyboost::PyBoostUtils::set_cur_stream_id(old_stream_id);

  MS_LOG(DEBUG) << "Run ${func_name} end";
  return py::reinterpret_steal<py::object>(tensor::Wrap(outputs));
}
