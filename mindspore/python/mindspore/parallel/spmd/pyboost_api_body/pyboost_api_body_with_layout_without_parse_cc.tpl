PYNATIVE_EXPORT PyObject* ${func_name}_Base(const PrimitivePtr &prim, PyObject* args) {
#ifndef ENABLE_TEST
  ${mark_side_effect}
  return WithLayoutInfer${suffix}(
    prim,
    [](const PrimitivePtr &p, const std::vector<ops::OP_DTYPE> &st${lambda_params}) {
      return ${func_name}_OP(p, st${lambda_args});
    },
    args);
#else
  py::object py_args = py::reinterpret_borrow<py::object>(args);
  py::object res = PyNativeAlgo::PyBoost::RunPyFunction(prim, py_args);
  return res.release().ptr();
#endif
}
