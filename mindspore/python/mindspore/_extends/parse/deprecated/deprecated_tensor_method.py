# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""Deprecated Tensor method"""

deprecated_tensor_method_map = {
    # 1 to

    # 2 masked_fill

    # 3 abs

    # 4 __abs__

    # 5 add

    # 6 all

    # 7 allclose

    # 8 any

    # 9 arctan2

    # 10 argmax
    "argmax": "deprecated_tensor_argmax",
    # 11 argmin

    # 12 argsort

    # 13 atan2

    # 14 bfloat16

    # 15 bmm

    # 16 bool

    # 17 broadcast_to

    # 18 byte

    # 19 ceil

    # 20 chunk

    # 21 clamp

    # 22 clip

    # 23 cos

    # 24 cumprod

    # 25 cumsum

    # 26 dim

    # 27 div

    # 28 divide

    # 29 eq

    # 30 erf

    # 31 exp

    # 32 expand

    # 33 expand_as

    # 34 flatten

    # 35 flip

    # 36 float

    # 37 floor

    # 38 gather
    "gather": "deprecated_tensor_gather",
    # 39 greater

    # 40 greater_equal

    # 41 gt

    # 42 half

    # 43 index_put

    # 44 index_select
    "index_select": "deprecated_tensor_index_select",
    # 45 int

    # 46 inverse

    # 47 is_contiguous

    # 48 isclose

    # 49 isfinite

    # 50 isnan

    # 51 item

    # 52 le

    # 53 less

    # 54 less_equal

    # 55 log

    # 56 log2

    # 57 logical_and
    "logical_and": "tensor_logical_and",
    # 58 logical_not

    # 59 logical_or
    "logical_or": "tensor_logical_or",

    # 60 long

    # 61 lt

    # 62 masked_fill

    # 63 masked_select

    # 64 matmul
    "matmul": "deprecated_tensor_matmul",
    # 65 max
    "max": "deprecated_tensor_max",
    # 66 maximum

    # 67 mean

    # 68 min
    "min": "deprecated_tensor_min",
    # 69 minimum

    # 70 mul

    # 71 nan_to_num

    # 72 narrow

    # 73 ne

    # 74 neg

    # 75 negative

    # 76 nonzero

    # 77 norm

    # 78 numel

    # 79 numpy

    # 80 outer

    # 81 permute

    # 82 pow

    # 83 prod
    "prod": "deprecated_tensor_prod",
    # 84 reciprocal

    # 85 remainder

    # 86 repeat

    # 87 repeat_interleave
    "repeat_interleave": "tensor_repeat_interleave",
    # 88 reshape

    # 89 round

    # 90 rsqrt

    # 91 scatter

    # 92 scatter_add

    # 93 select

    # 94 sigmoid

    # 95 sin

    # 96 size

    # 97 sort
    "sort": "deprecated_tensor_sort",
    # 98 split
    "split": "deprecated_tensor_split",
    # 99 sqrt

    # 100 square

    # 101 squeeze

    # 102 std

    # 103 sub

    # 104 sum

    # 105 swapaxes

    # 106 t

    # 107 tanh

    # 108 tile
    "tile": "deprecated_tensor_tile",
    # 109 tolist

    # 110 topk

    # 111 transpose

    # 112 tril
    "tril": "deprecated_tensor_tril",
    # 113 trunc

    # 114 type

    # 115 type_as

    # 116 unbind

    # 117 unfold

    # 118 unique

    # 119 unsqeeze

    # 120 view

    # 121 contiguous

    # 122 where
    "where": "deprecated_tensor_where",
    # 123 div_

    # 124 fill_

    # 125 floor_

    # 126 masked_fill_

    # 127 mul_

    # 128 normal_

    # 129 requires_grad_

    # 130 sub_

    # 131 uniform_

    # 132 absolute

    # 133 bincount

    # 134 diff

    # 135 double

    # 136 lcm

    # 137 mm

    # 138 ravel

    # 139 nelement

    # 140 stride

    # 141 indices

    # 142 view_as

    # 143 values

    # 144 index_copy

    # 145 element_size

    # 146 gcd

    # 147 isinf

    # 148 not_equal

    # 149 triu

    # 150 __eq__

    # 151

    # 152

    # 153

    # 154

    # 155

    # 156

    # 157

    # 158

    # 159

    # 160

    # 161

    # 162

}
