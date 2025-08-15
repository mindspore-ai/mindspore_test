PYNATIVE_EXPORT PyObject* ${func_name}_OP(const PrimitivePtr &prim, const std::vector<ops::OP_DTYPE>& source_type, ${input_args}) {
  MS_LOG(DEBUG) << "Run ${func_name} start";
  
  // AsyncStatus
  const auto &pynative_executor = pynative::PyNativeAlgo::Common::GetPyNativeExecutor();
  const auto& forward_executor = pynative_executor->forward_executor();
  const auto &device_target = forward_executor->GetCurrentDeviceTarget(prim);
  bool is_jit_compiling = forward_executor->is_jit_compiling();

  size_t custom_bprop_cell_count = pynative_executor->grad_executor()->custom_bprop_cell_count();
  bool requires_grad = pynative::GradState::Get().RequiresGrad();

  // TODO: Not support multi-thread yet.
  {
  GilReleaseWithCheck no_gil;
  runtime::Pipeline::Get().frontend_stage()->Wait();
  }

  // stub tensor to tensor.
  ${convert_stub}
  
  kernel::pyboost::OpRunStatus::Get().set_run_info(
      kernel::pyboost::OpStatus(true,
                                is_jit_compiling,
                                custom_bprop_cell_count,
                                device_target));
  kernel::pyboost::RequireGradGuard require_grad_guard(requires_grad);

  auto outputs = [&](){
    GilReleaseWithCheck no_gil;
    return kernel::pyboost::${operator_name}(${call_args});
  }();

  MS_LOG(DEBUG) << "Run ${func_name} end";
  return tensor::Wrap(outputs);
}
