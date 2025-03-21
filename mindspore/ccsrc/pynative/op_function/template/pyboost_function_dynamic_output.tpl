py::object PYNATIVE_EXPORT ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  auto op_run_info = PyNativeAlgo::PyBoost::Init(prim);
  op_run_info->signatures = ops::${op_def_name}.signatures_;
  op_run_info->source_type = source_type;

  // TODO: Not support multi-thread yet.
  {
    GilReleaseWithCheck no_gil;
    runtime::Pipeline::Get().frontend_stage()->Wait();
  }

  auto old_stream_id = kernel::pyboost::PyBoostUtils::cur_stream_id();
  kernel::pyboost::PyBoostUtils::set_cur_stream_id(op_run_info->base_op_run_info.stream_id);

  // stub tensor to tensor.
  ${convert_stub}

  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(op_run_info->async_status.disable_mix_precision,
                                op_run_info->async_status.is_jit_compiling,
                                op_run_info->async_status.custom_bprop_cell_count,
                                op_run_info->base_op_run_info.device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(op_run_info->requires_grad);

  auto outputs = [&](){
    // Do mixed precision and implicit cast
    static const std::vector<std::vector<size_t>> same_type_table{${same_type}};
    auto [${cast_args}] = PyNativeAlgo::PyBoost::SetPyBoostCastForInputs<${type_num}>(op_run_info, same_type_table, ${call_args});

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

py::object ${func_name}_Base(const PrimitivePtr &prim, const py::list &args) {
  #ifndef ENABLE_TEST
    static Converter converter(&ops::${op_def_name});
    converter.Parse(args);
    ${parser_body}
    return ${func_name}_OP(prim, converter.source_type(), ${op_args});
  #else
    return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
  #endif
}

py::object PYNATIVE_EXPORT ${func_name}(const py::args &args) {
  if (args.size() != kIndex2) {
  MS_LOG(EXCEPTION) << "Two args are needed by RunOp"
  << ", but got " << args.size();
  }
  const auto &prim = PyNativeAlgo::PyBoost::ConvertPrimitive(args[0]);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     prim->name(), false, true);
  return ${func_name}_Base(prim, args[1]);
}

class PYNATIVE_EXPORT ${class_name}PrimAdapter: public PrimitiveFunctionAdapter {
public:
${class_name}PrimAdapter() : PrimitiveFunctionAdapter() {}
~${class_name}PrimAdapter() = default;
std::string name() override { return "${class_name}"; }
py::object Call(const py::list &args) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp,
                                     "${class_name}", false, true);
  return ${func_name}_Base(prim::kPrim${class_name}, args);
}
};
