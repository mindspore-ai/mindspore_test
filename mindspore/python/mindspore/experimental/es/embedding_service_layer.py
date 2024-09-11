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
"""embedding service layer"""
import numpy as np

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore.ops.auto_generate import init_partition_map, init_embedding_hashmap, embedding_table_find_and_init,\
    embedding_table_find, fake_remote_lookup_uniqued
from mindspore.ops.auto_generate import EmbeddingTableImport, EmbeddingTableExport, \
    EmbeddingComputeVarImport, EmbeddingComputeVarExport, EmbeddingTableEvict, EmbeddingFeatureMappingV2, \
        EmbeddingFeatureMappingTableSize, EmbeddingFeatureMappingFind, EmbeddingFeatureMappingExport, \
            EmbeddingFeatureMappingFileSize, EmbeddingFeatureMappingImport, EmbeddingFeatureMappingInsert


class CounterFilter:
    """ Counter filter for embedding table. """
    def __init__(self, filter_freq, default_key_or_value, default_key=None, default_value=None):
        self.filter_freq = filter_freq
        self.default_key = default_key
        self.default_value = default_value
        self.default_key_or_value = default_key_or_value


def _get_slot_var_num(optimizer_mode):
    """ get slot var num by diff optimizer. """
    # adam, adamw, rmsprop include m and v, 2 slots; adagrad include accumulator, 1 slot; sgd include 0 slot
    if optimizer_mode == "adagrad":
        return 1
    if optimizer_mode == "sgd":
        return 0
    if optimizer_mode == "":
        return 0
    return 2


def _get_backward_float_params(optimizer_mode):
    """
    backward_float_params (Union[tuple[float], list[float]]):
        - when the backward_mode is 'adam', it means:
          [beta1_power, beta2_power, lr, beta1, beta2, epsilon]
        - when the backward_mode is 'ftrl', it means:
          [lr, lr_power, lambda1, lambda2]
        - when the backward_mode is 'adamw', it means:
          [beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon]
        - when the backward_mode is 'adagrad', it means [lr,]
    """
    if optimizer_mode == "adagrad" or optimizer_mode == "sgd":
        return [0.001]
    if optimizer_mode == "adam":
        return [0.9, 0.99, 0.001, 0.9, 0.999, 1e-08]
    if optimizer_mode == "ftrl":
        return [0.001, -0.5, 0.0, 0.0]
    if optimizer_mode == "rmsprop":
        return [0.001, 0.9, 0.1, 1e-08]
    # adamw
    return [0.9, 0.99, 0.001, 0.01, 0.9, 0.999, 1e-08]


class ESInitLayer(nn.Cell):
    """
    ESInitLayer.
    """
    def __init__(self, ps_num, ps_ids, train_mode, train_level, table_id, bucket_size, embedding_dim, slot_var_num,
                 es_initializer, filter_mode, optimizer, optimizer_params, max_feature_count, mode="train"):
        super(ESInitLayer, self).__init__()
        self.ps_num = ps_num
        self.ps_ids = ps_ids
        self.train_mode = train_mode
        self.train_level = train_level
        self.table_id = table_id
        self.bucket_size = bucket_size
        self.embedding_dim = embedding_dim
        self.es_initializer = es_initializer
        self.filter_mode = filter_mode
        self.optimizer_mode = optimizer if optimizer else ''
        self.optimizer_params = optimizer_params if optimizer_params else ()
        self.max_feature_count = max_feature_count

        self.ps_num_tensor = Tensor(self.ps_num, ms.int32)
        self.ps_ids_tensor = Tensor(self.ps_ids, ms.int32)
        self.table_id_tensor = Tensor(self.table_id, ms.int32)
        self.depend = ops.Depend()
        self.slot_var_num = _get_slot_var_num(self.optimizer_mode)
        if mode == "train":
            self.value_total_len = self.embedding_dim * (self.slot_var_num + 1) + 2
        else:
            self.value_total_len = self.embedding_dim * (self.slot_var_num + 1)
        self.filter_freq = None
        self.default_key = None
        self.default_value = None

    def construct(self):
        """
        ESInitLayer construct: init embedding hashmap
        """
        init_partition = init_partition_map(self.ps_num_tensor,
                                            self.ps_ids_tensor,
                                            _embedding_dim=self.embedding_dim,
                                            _max_key_num=self.max_feature_count,
                                            _ps_num=self.ps_num)
        depend = self.depend(self.table_id_tensor, init_partition)
        if self.train_mode:
            if self.train_level:
                return init_embedding_hashmap(table_id=depend,
                                              bucket_size=self.bucket_size,
                                              value_total_len=self.value_total_len,
                                              embedding_dim=self.embedding_dim,
                                              initializer_mode=self.es_initializer.initializer_mode,
                                              constant_value=self.es_initializer.constant_value,
                                              min=self.es_initializer.min,
                                              max=self.es_initializer.max,
                                              mu=self.es_initializer.mu,
                                              sigma=self.es_initializer.sigma,
                                              seed=self.es_initializer.seed,
                                              seed2=self.es_initializer.seed,
                                              filter_mode=self.filter_mode,
                                              optimizer_mode=self.optimizer_mode,
                                              optimizer_params=self.optimizer_params,
                                              _table_id=self.table_id)
            return init_embedding_hashmap(table_id=depend,
                                          bucket_size=self.bucket_size,
                                          value_total_len=self.value_total_len,
                                          embedding_dim=self.embedding_dim,
                                          initializer_mode=None, constant_value=None,
                                          min=None, max=None, mu=None, sigma=None,
                                          seed=None, seed2=None, filter_mode=self.filter_mode,
                                          optimizer_mode=self.optimizer_mode,
                                          optimizer_params=self.optimizer_params,
                                          _table_id=self.table_id)

        return init_embedding_hashmap(table_id=depend,
                                      value_total_len=self.value_total_len,
                                      embedding_dim=self.embedding_dim,
                                      bucket_size=self.bucket_size,
                                      filter_mode=self.filter_mode,
                                      optimizer_mode=self.optimizer_mode,
                                      optimizer_params=self.optimizer_params,
                                      _table_id=self.table_id)


class EsEmbeddingLookup(nn.Cell):
    r"""
    Look up a PS embedding.

    .. warning::
        This is an experimental EmbeddingService API that is subject to change.

    Args:
        table_id (int): The table id.
        es_initializer (EsInitializer): The EsInitialize object for PS embedding with table_id,
            which can be None when the inference is performed.
        embedding_dim (int): The embedding dim of keys for PS embedding with table_id.
        max_key_num (int): The num of keys when lookup.
        optimizer_mode (str): The type of optimizer. Default is ``None``.
        optimizer_params (tuple[float]): The parameters of optimizer. Default is ``None``.
        es_filter (CounterFilter): The option of counter filter for PS embedding with table_id. Default is ``None``.
        es_padding_key (PaddingParamsOption): The option of padding key for PS embedding with table_id.
            Default is ``None``.
        es_completion_key (CompletionKeyOption): The option of completion key for PS embedding with table_id.
            Default is ``None``.

    Inputs:
            - **keys** (Tensor): The keys of each feature in PS embedding.
            - **actual_keys_input** (Tensor): Tensor composed of all unique elements of keys.
            - **unique_indices** (Tensor): The index value of each element in keys to actual_keys_input .
            - **key_count** (Tensor): The count of each element in the actual_keys_input to keys.

    Supported Platforms:
        ``Atlas A2 training series products``
    """

    def __init__(self, table_id, es_initializer, embedding_dim, max_key_num, optimizer_mode=None,
                 optimizer_params=None, es_filter=None, es_padding_key=None, es_completion_key=None):
        super(EsEmbeddingLookup, self).__init__()
        self.cast = ops.cast
        self.reshape = ops.Reshape()

        self.table_id = Tensor(table_id, ms.int32)
        self._table_id = table_id
        self.es_initializer = es_initializer
        self.embedding_dim = embedding_dim
        self.optimizer_mode = optimizer_mode
        self.backward_float_params = _get_backward_float_params(self.optimizer_mode)
        self.max_key_num = max_key_num
        self.es_filter = es_filter

        self.slot_var_num = _get_slot_var_num(self.optimizer_mode)
        self.value_total_len = [self.embedding_dim[table_id] * (self.slot_var_num + 1) + 2] * len(embedding_dim)

        self.default_key_or_value = 1
        self.filter_freq = 0
        self.default_key = 0
        self.optimizer_params = optimizer_params

        if es_filter is not None:
            self.filter_mode = "counter"
            self.filter_freq = es_filter.filter_freq
            self.default_key_or_value = es_filter.default_key_or_value
            self.default_key = 0 if es_filter.default_key is None else es_filter.default_key
            self.default_value = 0.0
        else:
            self.filter_mode = "no_filter"
            self.filter_freq = 1
            self.default_key_or_value = 1
            self.default_key = 0
            self.default_value = 1.0

        self.global_step = 1
        if es_padding_key is not None:
            self.mask_zero = 0 if es_padding_key.mask_zero is None else int(es_padding_key.mask_zero)
            self.padding_key = es_padding_key.padding_key
            self.padding_key_mask = int(es_padding_key.mask)
        else:
            self.mask_zero = 0
            self.padding_key = 0
            self.padding_key_mask = 1
        if self.optimizer_mode in ["adam", "ftrl", "adagrad", "sgd", "rmsprop"]:
            self.backward_int_params = ([self.global_step], [self.mask_zero],
                                        [self.padding_key], [self.padding_key_mask])
        else:
            self.backward_int_params = ([self.global_step], [0], [0], [self.mask_zero],
                                        [self.padding_key], [self.padding_key_mask])

        if es_completion_key is not None:
            self.completion_key = es_completion_key.completion_key
            self.completion_key_mask = int(es_completion_key.mask)
        else:
            self.completion_key = 0
            self.completion_key_mask = 1

        self.b = Parameter(Tensor(0, ms.float32), name="b", requires_grad=True)
        self.max_grad_norm = Tensor([1.0], ms.float32)

    def construct(self, keys, actual_keys_input=None, unique_indices=None, key_count=None):
        """
        Using the corresponding query method to calculate the PS embedding for each key.
        """
        origin_shape = None
        if len(keys.shape) != 1:
            origin_shape = keys.shape
            keys = self.reshape(keys, (-1,))
        keys = self.cast(keys, ms.int64)
        use_host_unique = False
        use_counter_filter = 1 if self.filter_mode == "counter" else 0
        if (actual_keys_input is not None) and (unique_indices is not None):
            use_host_unique = True
            actual_keys_input = self.cast(actual_keys_input, ms.int64)
            unique_indices = self.cast(unique_indices, ms.int32)
        if use_host_unique:
            if not use_counter_filter:
                key_count = keys
        if self.training:
            if use_host_unique:
                output = fake_remote_lookup_uniqued(table_id=self.table_id, keys=keys,
                                                    actual_keys_num=actual_keys_input,
                                                    unique_indices=unique_indices, key_count=key_count,
                                                    max_grad_norm=self.max_grad_norm,
                                                    embedding_dim=self.embedding_dim,
                                                    initializer_mode=self.es_initializer.initializer_mode,
                                                    constant_value=self.es_initializer.constant_value,
                                                    min=self.es_initializer.min,
                                                    max=self.es_initializer.max,
                                                    mu=self.es_initializer.mu,
                                                    sigma=self.es_initializer.sigma,
                                                    seed=self.es_initializer.seed,
                                                    seed2=self.es_initializer.seed,
                                                    value_total_len=self.value_total_len,
                                                    filter_mode=self.filter_mode,
                                                    filter_freq=self.filter_freq,
                                                    default_key_or_value=self.default_key_or_value,
                                                    default_key=self.default_key,
                                                    default_value=self.default_value,
                                                    optimizer_mode=self.optimizer_mode,
                                                    optimizer_params=self.optimizer_params,
                                                    _max_key_num=self.max_key_num, _table_id=self._table_id,
                                                    _use_counter_filter=use_counter_filter,
                                                    backward_mode=self.optimizer_mode,
                                                    backward_int_params=self.backward_int_params,
                                                    backward_float_params=self.backward_float_params,
                                                    completion_key=self.completion_key,
                                                    completion_key_mask=self.completion_key_mask,
                                                    parameter=self.b
                                                    )
            else:
                output = embedding_table_find_and_init(self.table_id, keys,
                                                       max_grad_norm=self.max_grad_norm,
                                                       embedding_dim=self.embedding_dim,
                                                       initializer_mode=self.es_initializer.initializer_mode,
                                                       constant_value=self.es_initializer.constant_value,
                                                       min=self.es_initializer.min,
                                                       max=self.es_initializer.max,
                                                       mu=self.es_initializer.mu,
                                                       sigma=self.es_initializer.sigma,
                                                       seed=self.es_initializer.seed,
                                                       seed2=self.es_initializer.seed,
                                                       value_total_len=self.value_total_len,
                                                       filter_mode=self.filter_mode,
                                                       filter_freq=self.filter_freq,
                                                       default_key_or_value=self.default_key_or_value,
                                                       default_key=self.default_key,
                                                       default_value=self.default_value,
                                                       optimizer_mode=self.optimizer_mode,
                                                       optimizer_params=self.optimizer_params,
                                                       _max_key_num=self.max_key_num, _table_id=self._table_id,
                                                       _use_counter_filter=use_counter_filter,
                                                       backward_mode=self.optimizer_mode,
                                                       backward_int_params=self.backward_int_params,
                                                       backward_float_params=self.backward_float_params,
                                                       completion_key=self.completion_key,
                                                       completion_key_mask=self.completion_key_mask,
                                                       parameter=self.b)
        else:
            output = embedding_table_find(self.table_id, keys, embedding_dim=self.embedding_dim,
                                          default_value=self.default_value,
                                          _max_key_num=self.max_key_num, _table_id=self._table_id,
                                          _use_counter_filter=use_counter_filter)
        if origin_shape is not None:
            output = self.reshape(output, origin_shape + (-1,))
        return output


class ESEmbeddingCKPTExport(nn.Cell):
    """
    ESEmbeddingCKPTExport.
    """
    def __init__(self, embedding_dim_list, value_total_len_list, table_name_list, table_id_list,
                 file_path, steps_to_live_list):
        super(ESEmbeddingCKPTExport, self).__init__()
        self.embedding_table_export = EmbeddingTableExport(
            embedding_dim_list,
            value_total_len_list,
            table_name=table_name_list,
            steps_to_live_list=steps_to_live_list)
        self.embedding_compute_var_export = EmbeddingComputeVarExport(table_name_list)
        self.file_path = Tensor(np.array(file_path))
        self.ps_id_tensor = Tensor(0, ms.int32)
        self.table_id_tensor = Tensor(table_id_list, ms.int32)
        self.depend = ops.Depend()

    def construct(self, global_step):
        export_op1 = self.embedding_table_export(self.file_path, self.ps_id_tensor, self.table_id_tensor, global_step)
        z = self.depend(self.file_path, export_op1)
        export_op2 = self.embedding_compute_var_export(z, self.ps_id_tensor, self.table_id_tensor, global_step)
        return export_op2


class ESEmbeddingTableExport(nn.Cell):
    """
    ESEmbeddingTableExport.
    """
    def __init__(self, embedding_dim_list, value_total_len_list, table_name_list, table_id_list,
                 file_path, steps_to_live_list):
        super(ESEmbeddingTableExport, self).__init__()
        self.op = EmbeddingTableExport(
            embedding_dim_list,
            value_total_len_list,
            table_name=table_name_list,
            steps_to_live_list=steps_to_live_list,
            only_var_flag=True)
        self.file_path = Tensor(np.array(file_path))
        self.ps_id_tensor = Tensor(0, ms.int32)
        self.table_id_tensor = Tensor(table_id_list, ms.int32)

    def construct(self, global_step):
        y = self.op(self.file_path, self.ps_id_tensor, self.table_id_tensor, global_step)
        return y


class ESIncrementalEmbeddingTableExport(nn.Cell):
    """
    ESIncrementalEmbeddingTableExport.
    """
    def __init__(self, embedding_dim_list, value_total_len_list, table_name_list, table_id_list,
                 file_path, steps_to_live_list):
        super(ESIncrementalEmbeddingTableExport, self).__init__()
        self.op = EmbeddingTableExport(
            embedding_dim_list,
            value_total_len_list,
            table_name=table_name_list,
            steps_to_live_list=steps_to_live_list,
            export_mode="new",
            only_var_flag=True)
        self.file_path = Tensor(np.array(file_path))
        self.ps_id_tensor = Tensor(0, ms.int32)
        self.table_id_tensor = Tensor(table_id_list, ms.int32)

    def construct(self, global_step):
        y = self.op(self.file_path, self.ps_id_tensor, self.table_id_tensor, global_step)
        return y


class ESEmbeddingCKPTImport(nn.Cell):
    """
    ESEmbeddingCKPTImport.
    """
    def __init__(self, embedding_dim_list, value_total_len_list, table_name_list, table_id_list, file_path):
        super(ESEmbeddingCKPTImport, self).__init__()
        self.embedding_table_import = EmbeddingTableImport(
            embedding_dim_list,
            value_total_len_list,
            table_name=table_name_list)
        self.embedding_compute_var_import = EmbeddingComputeVarImport(table_name_list)
        self.file_path = Tensor(np.array(file_path))
        self.ps_id_tensor = Tensor(0, ms.int32)
        self.table_id_tensor = Tensor(table_id_list, ms.int32)
        self.depend = ops.Depend()

    def construct(self, global_step):
        export_op1 = self.embedding_table_import(self.file_path, self.ps_id_tensor, self.table_id_tensor, global_step)
        z = self.depend(self.file_path, export_op1)
        export_op2 = self.embedding_compute_var_import(z, self.ps_id_tensor, self.table_id_tensor, global_step)
        return export_op2


class ESEmbeddingTableImport(nn.Cell):
    """
    ESEmbeddingTableImport.
    """
    def __init__(self, embedding_dim_list, value_total_len_list, table_name_list, table_id_list, file_path):
        super(ESEmbeddingTableImport, self).__init__()
        self.op = EmbeddingTableImport(
            embedding_dim_list,
            value_total_len_list,
            table_name=table_name_list,
            only_var_flag=True)
        self.file_path = Tensor(np.array(file_path))
        self.ps_id_tensor = Tensor(0, ms.int32)
        self.table_id_tensor = Tensor(table_id_list, ms.int32)

    def construct(self, global_step):
        y = self.op(self.file_path, self.ps_id_tensor, self.table_id_tensor, global_step)
        return y


class ESEmbeddingTableEvict(nn.Cell):
    """
    ESEmbeddingTableEvict.
    """
    def __init__(self, var_handle, global_step, steps_to_live):
        super(ESEmbeddingTableEvict, self).__init__()
        self.op = EmbeddingTableEvict()
        self.var_handle = Tensor(var_handle, ms.int32)
        self.global_step = global_step
        self.steps_to_live = steps_to_live

    def construct(self):
        y = self.op(self.var_handle, self.global_step, self.steps_to_live)
        return y


class ESEmbeddingFeatureMappingExport(nn.Cell):
    """
    ESEmbeddingFeatureMappingExport.
    """
    def __init__(self, file_path, export_value, var, var_name, small_table_embedding_dim):
        super(ESEmbeddingFeatureMappingExport, self).__init__()
        self.embedding_feature_mapping_table_size = EmbeddingFeatureMappingTableSize()
        self.embedding_feature_mapping_find = EmbeddingFeatureMappingFind()
        self.embedding_feature_mapping_export = EmbeddingFeatureMappingExport()
        self.file_path = file_path
        self.export_value = export_value
        self.gather = ops.Gather()
        self.var = Tensor(var, ms.float32)
        self.var_name = Tensor(np.array([var_name]))
        self.small_table_embedding_dim = [small_table_embedding_dim]
        self.global_step = Tensor([-1], ms.int64)

    def construct(self):
        """
        ESEmbeddingFeatureMappingExport construct: export feature mapping for data_parallel embedding.
        """
        feature_size = self.embedding_feature_mapping_table_size(self.var_name)
        feature_id, offset_id = self.embedding_feature_mapping_find(self.var_name, feature_size, 1)
        values = self.gather(self.var, offset_id, 0)
        if self.export_value:
            embed_values = values
        else:
            embed_values = Tensor([0], ms.float32)
        feature_mapping_export = self.embedding_feature_mapping_export(self.file_path, self.var_name, self.global_step,
                                                                       embed_values, self.small_table_embedding_dim,
                                                                       [feature_id], [offset_id])
        return feature_mapping_export


class ESEmbeddingFeatureMappingImport(nn.Cell):
    """
    ESEmbeddingFeatureMappingImport.
    """
    def __init__(self, file_path, small_table_name, small_table_embedding_dim, only_offset_flag):
        super(ESEmbeddingFeatureMappingImport, self).__init__()
        self.embedding_feature_mapping_file_size = EmbeddingFeatureMappingFileSize()
        self.embedding_feature_mapping_import = EmbeddingFeatureMappingImport()
        self.embedding_feature_mapping_insert = EmbeddingFeatureMappingInsert()
        self.file_path = file_path
        self.small_table_name = Tensor(np.array([small_table_name]))
        self.small_table_embedding_dim = [small_table_embedding_dim]
        self.only_offset_flag = only_offset_flag
        self.global_step = Tensor([-1], ms.int64)

    def construct(self):
        """
        ESEmbeddingFeatureMappingImport construct: import feature mapping for data_parallel embedding.
        """
        feature_size = self.embedding_feature_mapping_file_size(self.file_path,
                                                                self.small_table_name,
                                                                self.global_step,
                                                                self.small_table_embedding_dim,
                                                                self.only_offset_flag)
        feature_id, offset_id = self.embedding_feature_mapping_import(self.file_path,
                                                                      self.small_table_name,
                                                                      feature_size, self.global_step,
                                                                      self.small_table_embedding_dim,
                                                                      self.only_offset_flag, 1)
        feature_mapping_insert = self.embedding_feature_mapping_insert(self.small_table_name, 1,
                                                                       [feature_id], [offset_id])
        return feature_mapping_insert


class ESEmbeddingSmallTableLookup(nn.Cell):
    r"""
    Look up a data_parallel embedding.

    .. warning::
        This is an experimental EmbeddingService API that is subject to change.

    Args:
        name (str): The data_parallel embedding name.
        rank_id (int): The rank id when look up data_parallel embedding key.
        rank_size (int): The rank size when look up data_parallel embedding key.
        small_table_to_variable (dict[str, parameter]): The dict to restore data_parallel embedding information:
            key is table name, value is parameter.

    Inputs:
        - **ids_list** (Tensor) - The keys of each feature in data_parallel embedding.

    Supported Platforms:
        ``Atlas A2 training series products``
    """

    def __init__(self, name, rank_id, rank_size, small_table_to_variable):
        super(ESEmbeddingSmallTableLookup, self).__init__()
        self.small_table_to_variable = small_table_to_variable[name]
        self.small_table_to_variable.feature_name = name
        self.allgather = ops.AllGather()
        self.gather = ops.Gather()
        self.embedding_feature_mapping_v2 = EmbeddingFeatureMappingV2()
        self.name = name
        self.rank_id = rank_id
        self.rank_size = rank_size

    def construct(self, ids_list):
        """
        Using the EmbeddingFeatureMappingV2 method to mapping hash key to non hash key, and then get embedding value.
        """
        hash_key_shape = ids_list.shape
        if self.rank_size > 1 and (hash_key_shape[0] is not None):
            hash_key = ops.stop_gradient(self.allgather(ids_list))
            non_hash_key = self.embedding_feature_mapping_v2(self.name, hash_key, [1], [1])
            recovery_matrix = []
            for i in range(hash_key_shape[0]):
                recovery_matrix.append(self.rank_id * hash_key_shape[0] + i)
            local_non_hash_keys = self.gather(non_hash_key, Tensor(recovery_matrix), 0)
        else:
            hash_key = ids_list
            local_non_hash_keys = self.embedding_feature_mapping_v2(self.name, hash_key, [1], [1])

        embedding = self.gather(self.small_table_to_variable, local_non_hash_keys, 0)
        return embedding
