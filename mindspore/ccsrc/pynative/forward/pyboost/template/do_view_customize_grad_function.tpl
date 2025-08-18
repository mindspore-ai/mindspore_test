void DoGrad${class_name}(${grad_args_with_type}) {
  static bool bprop_expander = ${bprop_expander};
  static bool non_differentiable = ${non_differentiable};
  if (!bprop_expander || non_differentiable) {
    return;
  }

  // DoGrad${class_name}Impl is customized, which is implemented in
  // mindspore/ccsrc/pynative/forward/pyboost/customize/grad_impl.h
  DoGrad${class_name}Impl(output, ${grad_input_args_with_optional});
}