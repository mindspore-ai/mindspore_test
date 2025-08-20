# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
"""shard"""

import mindspore as ms
from mindspore._c_expression import TensorTransform
from mindspore import log as logger
from mindspore.communication import get_rank, create_group
from mindspore.parallel.redistribute_infer import RedistributionOperatorInfer

_tensor_transform = TensorTransform.get_instance()


_REDISTRIBUTION_GROUP_CACHE = []


def _get_comm_group(rank_list):
    """_get_comm_group"""
    global _REDISTRIBUTION_GROUP_CACHE
    hash_str_rank_list = '-'.join([str(rank) for rank in rank_list])
    group_name = f"{len(rank_list)}-{hash_str_rank_list}"
    if group_name not in _REDISTRIBUTION_GROUP_CACHE:
        logger.warning(f"Create hccl comm group {group_name} for rank list {rank_list}")
        create_group(group_name, rank_list)
        _REDISTRIBUTION_GROUP_CACHE.append(group_name)
    return group_name



def _construct_layout_tuple_for_transform_operator_list(from_layout, to_layout, from_full_shape):
    """_construct_layout_tuple_for_transform_operator_list"""
    from_layout_dict = from_layout.to_dict()
    to_layout_dict = to_layout.to_dict()
    from_layout_tuple = (from_layout_dict["device_matrix"], from_layout_dict["tensor_map"], list(from_full_shape))
    to_layout_tuple = (to_layout_dict["device_matrix"], to_layout_dict["tensor_map"], list(from_full_shape))  # TODO: 考虑reshape的场景
    return from_layout_tuple, to_layout_tuple


def print_transform_operator(transform_operator_list):
    """print_transform_operator"""
    logger.warning(f"Transform operator list size: {len(transform_operator_list)}")
    for i, transform_operator in enumerate(transform_operator_list):
        logger.warning(f"[{i}] {transform_operator[0]}: {transform_operator[1]}")


class TensorRedistribution:
    """
    TensorRedistribution.
    """
    def __init__(self):
        self.is_init = False
        self.rank_list = None # rank_list for current stage
        self.rank_id = None # current rank_lid
        self._transform_cache = {}
        self._construct_op_operator = {
            "Reshape": self._construct_reshape,
            "AllConcat": self._construct_all_concat,
            "StridedSlice": self._construct_strided_slice,
            "all_concat": self._construct_all_concat_new,
            "all_split": self._construct_all_split,
            "all_to_all": self._construct_all_to_all
        }

    def _construct_reshape(self, x, *args):
        """args: (*shape)"""
        return x.view(args)


    def _construct_all_concat(self, x, *args):
        """args: (*rank_list, concat_dim)"""
        rank_list = list(args[0:-1])
        concat_dim = args[-1]
        group = _get_comm_group(rank_list)
        empty_tensor = [ms.mint.empty(x.shape, dtype=x.dtype) for _ in rank_list]
        x = x.contiguous()
        _ = ms.mint.distributed.all_gather(empty_tensor, x, group)
        return ms.mint.concat(empty_tensor, concat_dim)


    def _construct_strided_slice(self, x, *args):
        """args: (begin, end, strides)"""
        dims = len(args) // 3
        return ms.ops.strided_slice(x, args[0: dims], args[dims: 2*dims], args[2*dims:])

    def _construct_all_concat_new(self, x, *args):
        """args: (concat_dim, concat_size, group)"""
        rank_list = list(args[2])
        concat_dim = args[0]
        group = _get_comm_group(rank_list)
        empty_tensor = [ms.mint.empty(x.shape, dtype=x.dtype) for _ in rank_list]
        x = x.contiguous()
        _ = ms.mint.distributed.all_gather(empty_tensor, x, group)
        return ms.mint.concat(empty_tensor, concat_dim)

    def _construct_all_split(self, x, *args):
        """args: (split_dim, split_size, group)"""
        rank_list = list(args[2])
        split_dim = args[0]
        split_size = x.shape[split_dim] // args[1]
        idx = rank_list.index(self.rank_id)
        return ms.mint.split(x, split_size, split_dim)[idx]

    def _construct_all_to_all(self, x, *args):
        """args: (split_dim, concat_dim, permute_size, group)"""
        rank_list = list(args[3])
        split_dim = args[0]
        concat_dim = args[1]
        permute_size = x.shape[split_dim] // args[2]
        group = _get_comm_group(rank_list)
        send_tensor = ms.mint.split(x, permute_size, split_dim)
        recv_tensor = [ms.mint.zeros_like(send_tensor[0]) for _ in rank_list]
        _ = ms.mint.distributed.all_to_all(recv_tensor, send_tensor, group)
        return ms.mint.concat(recv_tensor, concat_dim)

    def _apply_eazy_redistribute(self, src_layout, dst_layout):
        """_apply_eazy_redistribute"""
        if (src_layout.device_matrix != dst_layout.device_matrix or
                src_layout.rank_list != dst_layout.rank_list):
            return False

        tensor_map_size = len(src_layout.tensor_map)
        if len(dst_layout.tensor_map) != tensor_map_size:
            return False
        return True

    def _redistribution_without_shape(self, local_x, src_layout, dst_layout, key):
        """_redistribution_without_shape"""
        inferrer = RedistributionOperatorInfer(
            dev_mat=src_layout.device_matrix,
            in_tensor_map=list(src_layout.tensor_map),
            out_tensor_map=list(dst_layout.tensor_map)
        )
        op_list = inferrer.InferOpsList(self.rank_id, self.rank_list)
        self._transform_cache[key] = op_list
        for op in op_list:
            local_x = self._construct_op_operator[op[0]](local_x, *op[1])
        return local_x


    def redistribution(self, x, to_layout):
        """ tensor redistribution """
        from_layout = x.layout

        if not self.is_init:
            self.rank_id = get_rank()
            self.rank_list = from_layout.rank_list
            self.is_init = True
        if self.rank_list != to_layout.rank_list:
            raise ValueError(f"The from_layout rank list: {self.rank_list} is not equal to "
                             f"to_layout rank list: {to_layout.rank_list}")
        key = from_layout.compact_str + to_layout.compact_str +  str(self.rank_id)
        if key in self._transform_cache:
            x = x.to_local()
            transform_operator_list = self._transform_cache[key]
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
            x = x.local_to_global(to_layout)
            return x

        full_shape = x.shape
        key_and_shape = key + str(full_shape)
        x = x.to_local()
        if key_and_shape in self._transform_cache:
            transform_operator_list = self._transform_cache[key_and_shape]
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
            x = x.local_to_global(to_layout)
            return x

        if self._apply_eazy_redistribute(from_layout, to_layout):
            x = self._redistribution_without_shape(x, from_layout, to_layout, key)
        else:
            transform_operator_list = self._infer_transform_operator_list(from_layout, to_layout,
                                                                          full_shape, key_and_shape)
            for transform_operator in transform_operator_list:
                x = self._construct_op_operator[transform_operator[0]](x, *transform_operator[1])
        x = x.local_to_global(to_layout)
        return x

    def _infer_transform_operator_list(self, from_layout, to_layout, from_full_shape, key):
        """infer transform operator list"""
        from_layout_tuple, to_layout_tuple = \
            _construct_layout_tuple_for_transform_operator_list(from_layout, to_layout, from_full_shape)
        self._transform_cache[key] = \
            _tensor_transform.transform_tensor_sharding(from_layout_tuple, to_layout_tuple, self.rank_list,
                                                        False, self.rank_id)
        return self._transform_cache[key]


_tensor_redistribution = TensorRedistribution()
