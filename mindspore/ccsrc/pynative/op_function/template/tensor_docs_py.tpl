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
"""Add docstrings to Tensor functions"""
from mindspore.common.tensor import Tensor
from mindspore._c_expression import _add_docstr as add_docstr


def attach_docstr(method, docstr):
    try:
        add_docstr(getattr(Tensor, method), docstr)
    except Exception as e:
        raise AttributeError(
            f"Failed to attach docstring to Tensor.{method}.\n"
            f"Please check if there is a duplicate Tensor.{method} in tensor.py."
        )

${add_doc_statements}
