void DoGrad${class_name}Inner(${inner_grad_args_with_type}, const ValuePtr &output_value) {
  MS_LOG(DEBUG) << "In DoGrad${class_name}";
  static auto op_type = kernel::pyboost::GetOpTypeFromOpdef(ops::g${class_name});
  auto grad_info = std::make_shared<OpGradInfo>(op_type,
                                                prim::kPrim${class_name},
                                                std::vector<ValuePtr>{${grad_input_args}},
                                                output_value);
  AutoGradUtil::SetInfer${is_multi}OutputToGrad(grad_info, op);
  PyNativeAlgo::PyBoost::DoGrad(op, grad_info, GetAsyncStatus());
  MS_LOG(DEBUG) << "Out DoGrad${class_name}";
}

void DoGrad${class_name}(${grad_args_with_type}) {
  static bool is_inplace_op = kernel::pyboost::GetOpTypeFromOpdef(ops::${op_def_name}) == OperatorType::kInplaceOp;
  static bool bprop_expander = ${bprop_expander};
  static bool non_differentiable = ${non_differentiable};
  if (!bprop_expander || non_differentiable) {
    return;
  }

  bool require_grad = NeedAutoGrad();

  auto output_value = AutoGradUtil::Make${is_multi}Output(require_grad, op);

  if (require_grad) {
    DoGrad${class_name}Inner(op, ${grad_input_args_with_optional}, output_value);
  } else if (is_inplace_op) {
    PyNativeAlgo::PyBoost::BumpVersionAsync(op->outputs()[0]->version());
  }
}
