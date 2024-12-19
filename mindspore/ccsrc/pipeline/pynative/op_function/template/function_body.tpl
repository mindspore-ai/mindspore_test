${return_type} ${op_name}(${input_args_with_type}) {
  MS_LOG(DEBUG) << "In ${op_name} function";

  const auto &device_target = GetDeviceTarget();
  auto op = CREATE_PYBOOST_OP(${class_name}, device_target);
  ${clone_func}
  auto output = op->Call(${input_args});

  static auto ${op_name}_grad_func = AutoGradFactory::Get().ops_auto_grad_registers().${class_name}GradFuncObj;
  ${op_name}_grad_func(op, ${input_args});
  MS_LOG(DEBUG) << "Out ${op_name} function";
  OpRunStatus::Get().SetLastOp(op);
  return output;
}
