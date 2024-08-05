# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Registry the relation."""

from __future__ import absolute_import
import os
from mindspore import context
from mindspore._c_expression import Tensor as Tensor_


class Registry:
    """Used for tensor operator registration"""

    def __init__(self):
        self._tensor_method_map = {}
        self._mint_functions = []

    def register(self, obj_str, obj, is_mint=False):
        """ Register the relation."""
        if not isinstance(obj_str, str):
            raise TypeError("key for tensor registry must be string.")
        if is_mint:
            if os.environ.get("MS_TENSOR_METHOD_BOOST") == '1' and context.get_context("device_target") == 'Ascend':
                self._tensor_method_map[obj_str] = obj
                self._mint_functions.append(obj_str)
        else:
            if obj_str not in self._mint_functions:
                self._tensor_method_map[obj_str] = obj

    def get(self, obj_str):
        """Get property if obj is not vm_compare"""

        if not isinstance(obj_str, str):
            raise TypeError("key for tensor registry must be string.")
        if Tensor_._is_test_stub() is True:  # pylint: disable=W0212
            def wrap(*args):
                new_args = list(args)
                new_args.append(obj_str)
                return self._tensor_method_map["vm_compare"](*new_args)

            obj = wrap
        else:
            obj = self._tensor_method_map[obj_str]
        return obj


tensor_operator_registry = Registry()
