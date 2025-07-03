class ${class_name}Register {
 public:
  ${class_name}Register() {
    tensor::TensorPyboostMethodRegister::Register(tensor::TensorPyboostMethod::k${class_name}Reg, [](const py::list &args) {
      static ${class_name}PrimAdapter ${op_name}_prim;
      return ${op_name}_prim.Call(args);
    });
    MS_LOG(DEBUG) << "Register tensor method ${op_name}";
  }
};

static ${class_name}Register ${op_name}_reg;

