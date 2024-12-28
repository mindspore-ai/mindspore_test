# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Op precision interfaces."""

from mindspore._checkparam import args_type_check
try:
    from mindspore._c_expression import GPUOpPrecisionConf
except ImportError:
    pass


@args_type_check(value=bool)
def matmul_allow_tf32(value):
    """
    Whether to convert FP32 to TF32 for Matmul operators.
    For detailed information, please refer to `CUBLAS_COMPUTE_32F_FAST_TF32 <https://docs.nvidia.com/cuda/cublas/index.html>`_.

    Args:
        value (bool): Whether to convert FP32 to TF32 for Matmul operators. If not configured, the framework
            defaults to ``False``.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.op_precision.matmul_allow_tf32(True)
    """
    GPUOpPrecisionConf.get_instance().matmul_allow_tf32(value)


@args_type_check(value=bool)
def conv_allow_tf32(value):
    """
    Whether to convert FP32 to TF32 for Conv operators.
    For detailed information, please refer to `CUBLAS_COMPUTE_32F_FAST_TF32 <https://docs.nvidia.com/cuda/cublas/index.html>`_.

    Args:
        value (bool): Whether to convert FP32 to HF32 for Conv operators. If not configured, the framework defaults
            to ``True``.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.gpu.op_precision.conv_allow_tf32(False)
    """
    GPUOpPrecisionConf.get_instance().conv_allow_tf32(value)
