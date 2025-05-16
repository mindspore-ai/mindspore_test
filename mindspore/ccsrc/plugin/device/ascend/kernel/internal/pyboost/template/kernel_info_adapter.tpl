class ${op_name}KernelInfoAdapter {
 public:
  ${op_name}KernelInfoAdapter() = default;
  virtual ~${op_name}KernelInfoAdapter() = default;

  virtual void InternalKernelCall(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type}) = 0;
  virtual void CreateKernelInfo(const std::string &kernel_name) = 0;
};