py::function fn;
py::object res;
py::object self_new = py::reinterpret_borrow<py::object>(self);
py::args py_args_new = py::reinterpret_borrow<py::args>(py_args);
py::kwargs py_kwargs_new = py::reinterpret_borrow<py::kwargs>(py_kwargs);
switch (parse_args.GetOvertLoadIndex()) {
  ${ut_dispatch_cases}
}
return res.release().ptr();