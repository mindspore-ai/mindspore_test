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

"""Mint adaptor."""

from __future__ import absolute_import
import os
from mindspore.common._register_for_tensor import tensor_operator_registry_for_mint

def repeat_interleave_mint(orig_fn):
    """
    repeat_interleave warpper.
    For details, please refer to :func:`mindspore.ops.repeat_interleave_ext`.
    """
    def wrapper(self, *args, **kwargs):
        if os.environ.get("MS_TENSOR_API_ENABLE_MINT") == "1":
            print("--> repeat interleave use mint.")
            return tensor_operator_registry_for_mint.get('repeat_interleave')(self, *args, **kwargs)
        return orig_fn(self, *args, **kwargs)
    return wrapper

def isnan_mint(orig_fn):
    """
    isnan warpper.
    """
    def wrapper(self, *args, **kwargs):
        if os.environ.get("MS_TENSOR_API_ENABLE_MINT") == "1":
            print("--> isnan use mint.")
            return tensor_operator_registry_for_mint.get('ne')(self, self, **kwargs)
        return orig_fn(self, *args, **kwargs)
    return wrapper


def add_mint(add):
    def wrapper(self, other, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            if "alpha" not in kwargs:
                kwargs["alpha"] = 1
            return tensor_operator_registry_for_mint.get('add')(self, other, **kwargs)
        return add(self, other, **kwargs)
    return wrapper


def flatten_mint(flatten):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            if "start_dim" not in kwargs:
                kwargs["start_dim"] = 0
            if "end_dim" not in kwargs:
                kwargs["end_dim"] = -1
            return tensor_operator_registry_for_mint.get('flatten')(self, **kwargs)
        return flatten(self, *args, **kwargs)
    return wrapper

def item_mint(fn):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            return tensor_operator_registry_for_mint.get('item')(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return wrapper

def max_mint(fn):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            return tensor_operator_registry_for_mint.get('max')(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return wrapper

def mean_mint(fn):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            return tensor_operator_registry_for_mint.get('mean')(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return wrapper

def min_mint(fn):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            return tensor_operator_registry_for_mint.get('min')(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return wrapper

def split_mint(split):
    def wrapper(self, *args, **kwargs):
        if os.environ.get("MS_TENSOR_API_ENABLE_MINT") == '1':
            print("--> use mint")
            if len(args) > 1 and isinstance(args[1], int):
                return tensor_operator_registry_for_mint.get('split')(self, args[0], args[1])
            else:
                return tensor_operator_registry_for_mint.get('split')(self, *args, **kwargs)
        return split(self, *args, **kwargs)
    return wrapper

def sub_mint(sub):
    def wrapper(self, *args, **kwargs):
        if os.environ.get("MS_TENSOR_API_ENABLE_MINT") == '1':
            print("--> use mint")
            if len(args) > 1 and isinstance(args[1], int):
                return tensor_operator_registry_for_mint.get('sub')(self, args[0], args[1])
            else:
                return tensor_operator_registry_for_mint.get('sub')(self, *args, **kwargs)
        return sub(self, *args, **kwargs)
    return wrapper
