  ProfileTrackerTask();
  if (IsOpPluginKernel(op_name())) {
    outputs_ = PyboostLaunchOpPluginKernel(get_op(), ${call_args});
    return ${return_values};
  }
  ${customize_func}(get_op(), ${call_args});
  get_op()->CreateOutputSimpleInfo();
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});
  return ${return_values};