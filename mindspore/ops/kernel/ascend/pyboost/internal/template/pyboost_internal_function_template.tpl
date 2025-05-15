void internal_${operator_name}(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type}) {
  if (!is_plugin_loaded) {
    LoadPlugin();
  }
  static std::shared_ptr<${op_name}KernelInfoAdapter> kernel_info_adapter = nullptr;
  if (kernel_info_adapter == nullptr) {
    kernel_info_adapter = Factory<${op_name}KernelInfoAdapter>::Instance().Create("${op_name}");
  }
  std::shared_ptr<${op_name}KernelInfoAdapter> kernel_info_adapter_ptr = kernel_info_adapter;
  pyboost::PyBoostUtils::DispatchRun(
  std::make_shared<runtime::PyBoostDeviceTask>(
    [op, kernel_info_adapter_ptr, ${call_args_after_convert}]() {
      kernel_info_adapter_ptr->InternalKernelCall(op, ${call_args_after_convert});
    }
  ));
}
