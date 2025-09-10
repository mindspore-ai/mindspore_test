py::object PYNATIVE_EXPORT ${func_name}_Base(const PrimitivePtr &prim, const py::list &args) {
#ifndef ENABLE_TEST
  ${mark_side_effect}
  static Converter converter(&ops::g${class_name});
  converter.Parse(args);
  ${parser_body}
  auto source_type = converter.source_type();

  return WithLayoutInfer(
    prim,
    [](const PrimitivePtr &p, const std::vector<ops::OP_DTYPE> &st${lambda_params}) {
      return ${func_name}_OP(p, st${lambda_args});
    },
    args,
    prim, source_type${forward_args}
  );
#else
  return PyNativeAlgo::PyBoost::RunPyFunction(prim, args);
#endif
}

