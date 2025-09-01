void DoGrad${class_name}Inner(${inner_grad_args_with_type}) {
  MS_LOG(DEBUG) << "In DoGrad${class_name}";
  static auto op_type = kernel::pyboost::GetOpTypeFromOpdef(ops::g${class_name});
  auto grad_info = std::make_shared<OpGradInfo>(op_type,
                                                prim::kPrim${class_name},
                                                std::vector<ValuePtr>{${grad_input_args}},
                                                output_value);
  PyNativeAlgo::PyBoost::DoGrad(grad_info, GetAsyncStatus());
  MS_LOG(DEBUG) << "Out DoGrad${class_name}";
}

void DoGrad${class_name}(${grad_args_with_type}) {
  static bool bprop_expander = ${bprop_expander};
  static bool non_differentiable = ${non_differentiable};
  if (!bprop_expander || non_differentiable) {
    return;
  }

  bool require_grad = NeedAutoGrad();

  auto output_value = AutoGradUtil::MakeOutput(require_grad, output${view_arg});

  if (require_grad) {
    ${convert_basic_to_value}
    DoGrad${class_name}Inner(output_value, ${grad_input_args_with_optional});
  }
}
