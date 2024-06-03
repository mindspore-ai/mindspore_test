import os

# op_def
OP_DEF_AUTO_GENERATE_PATH = "op_def/auto_generate"
MS_OP_DEF_AUTO_GENERATE_PATH = "mindspore/ops/op_def/auto_generate"
YAML_PATH = "op_def/yaml"
MS_YAML_PATH = "mindspore/ops/" + YAML_PATH
PY_AUTO_GEN_PATH = "mindspore/python/mindspore/ops/auto_generate"
PY_OPS_GEN_PATH = "mindspore/python/mindspore/ops_generate"

# infer
MS_OPS_FUNC_IMPL_PATH = "mindspore/ops/infer/ops_func_impl"

# view
MS_OPS_VIEW_PATH = "mindspore/ops/view"

# kernel
MS_OPS_KERNEL_PATH = "mindspore/ops/kernel"
MS_COMMON_PYBOOST_KERNEL_PATH = os.path.join(MS_OPS_KERNEL_PATH,  "common/pyboost")