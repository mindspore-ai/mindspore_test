  ProfileTrackerTask();
  ${check_expression}
  ${customize_func}(get_op(), ${call_args});
  get_op()->CreateOutputSimpleInfo();
  ProfileTrackerInput(${call_args});
  ProfileTrackerOutput(${return_values});
  return ${return_values};
