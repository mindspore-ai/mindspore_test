class Internal${op_name}KernelInfoAdapter : public ${op_name}KernelInfoAdapter {
 public:
  Internal${op_name}KernelInfoAdapter() = default;
  ~Internal${op_name}KernelInfoAdapter() override = default;

  void InternalKernelCall(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type}) override;
  void CreateKernelInfo(const std::string &kernel_name) override;

 private:
  std::shared_ptr<${op_name}> kernel_info_{nullptr};
};