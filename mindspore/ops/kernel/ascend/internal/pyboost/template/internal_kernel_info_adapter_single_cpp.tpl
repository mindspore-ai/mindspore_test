void Internal${op_name}KernelInfoAdapter::CreateKernelInfo(const std::string &kernel_name) {
  if (kernel_info_ != nullptr) {
    return;
  }
  if (Factory<InternalKernelInfo>::Instance().IsRegistered(kernel_name)) {
    MS_LOG(INFO) << "Supported by Internal Op: " << kernel_name;
    kernel_info_ =
      std::dynamic_pointer_cast<${op_name}>(Factory<InternalKernelInfo>::Instance().Create(kernel_name));
  }
  if (kernel_info_ == nullptr) {
    MS_LOG(EXCEPTION) << "Internal can't find Op[" << kernel_name << "]";
    return;
  }
}

void Internal${op_name}KernelInfoAdapter::InternalKernelCall(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type}) {
  auto kernel_name = op->primitive()->name();
  CreateKernelInfo(kernel_name);
  auto op_key = CalcInternalOpApiHash(kernel_name, ${call_args_after_convert});
  auto tiling_key = CalcInternalOpTilingHash(kernel_name, ${call_args_after_convert});
  kernel_info_->Call(op, op_key, tiling_key, ${call_args_after_convert});
}