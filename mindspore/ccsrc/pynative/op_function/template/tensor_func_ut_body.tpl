MS_LOG(INFO) << "Callback python method in UT: ${py_method}";
py::object self_new = py::reinterpret_borrow<py::object>(self);
py::args py_args_new = py::reinterpret_borrow<py::args>(py_args);
py::kwargs py_kwargs_new = py::reinterpret_borrow<py::kwargs>(py_kwargs);
py::function fn = python_adapter::GetPyFn("mindspore.ops.tensor_method", "${py_method}");
py::object res = fn(self_new, *py_args_new, **py_kwargs_new);
return res.release().ptr();