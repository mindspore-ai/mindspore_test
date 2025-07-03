void PYBOOST_API ${func_name}(OpRunnerInfo* op_runner_info, VectorRef *op_outputs) {
  MS_EXCEPTION_IF_NULL(op_runner_info);
  ${convert_body}
  kernel::pyboost::OpRunStatus::Get().set_run_info(
        kernel::pyboost::OpStatus(true, false, 0, op_runner_info->device_target));
  auto outputs = kernel::pyboost::${operator_name}(${call_args});
  auto op = kernel::pyboost::OpRunStatus::Get().GetLastOp();
  if (op->output_value_simple_info() != nullptr) {
    op_runner_info->output_value_simple_info = op->output_value_simple_info();
  } else {
    MS_EXCEPTION_IF_NULL(op->output_abs());
    op_runner_info->output_abs = op->output_abs();
  }
  MS_EXCEPTION_IF_NULL(op_outputs);
  (void)std::transform(op->outputs().begin(), op->outputs().end(), std::back_inserter(*op_outputs),
                       [] (const auto &item) {return item;});
}
