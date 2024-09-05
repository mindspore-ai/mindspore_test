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
"""embedding service"""
import json
import os
import math
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore.experimental.es.embedding_service_layer import ESInitLayer
from mindspore.common.initializer import Uniform, TruncatedNormal, Constant
from mindspore.common.initializer import initializer as ms_initializer
from mindspore.experimental.es.embedding_service_layer import ESEmbeddingTableImport, ESEmbeddingTableExport, \
    ESIncrementalEmbeddingTableExport, ESEmbeddingCKPTImport, ESEmbeddingCKPTExport, ESEmbeddingTableEvict, \
    ESEmbeddingFeatureMappingExport, ESEmbeddingFeatureMappingImport

_INT32_MAX_VALUE = 2147483647


class CounterFilter:
    """ Counter filter for embedding table. """
    def __init__(self, filter_freq, default_key_or_value, default_key=None, default_value=None):
        self.filter_freq = filter_freq
        self.default_key = default_key
        self.default_value = default_value
        self.default_key_or_value = default_key_or_value


class PaddingParamsOption:
    """ padding key option for embedding service table. """
    def __init__(self, padding_key=None,
                 mask=True,
                 mask_zero=False):
        self.padding_key = padding_key
        self.mask = mask
        self.mask_zero = mask_zero


class CompletionKeyOption:
    """ completion key option for embedding service table. """
    def __init__(self, completion_key=None, mask=1):
        self.completion_key = completion_key
        self.mask = mask


class EvictOption:
    """ Evict option for embedding table. """
    def __init__(self, steps_to_live):
        self.steps_to_live = steps_to_live


class EmbeddingVariableOption:
    """ option for embedding service table. """
    def __init__(self, filter_option=None,
                 padding_option=None,
                 evict_option=None,
                 completion_option=None,
                 storage_option=None,
                 feature_freezing_option=None,
                 communication_option=None):
        self.filter_option = filter_option
        self.padding_option = padding_option
        self.evict_option = evict_option
        self.completion_option = completion_option
        self.storage_option = storage_option
        self.feature_freezing_option = feature_freezing_option
        self.communication_option = communication_option


class EsInitializer:
    """Initializer for embedding service table."""
    def __init__(self, initializer_mode, min_scale=-0.01, max_scale=0.01,
                 constant_value=1.0, mu=0.0, sigma=1.0, seed=0):
        self.initializer_mode = initializer_mode
        self.min = min_scale
        self.max = max_scale
        self.constant_value = constant_value
        self.mu = mu
        self.sigma = sigma
        self.seed = seed


def check_common_init_params(name, init_vocabulary_size, embedding_dim, embedding_type):
    """
    Check init parameters.
    """
    if (name is None) or (init_vocabulary_size is None) or (embedding_dim is None):
        raise ValueError("table name, init_vocabulary_size and embedding_dim can not be None.")
    if not isinstance(name, str):
        raise TypeError("embedding table name must be string.")
    if (not isinstance(init_vocabulary_size, int)) or (not isinstance(embedding_dim, int)):
        raise ValueError("init_vocabulary_size and embedding_dim must be int.")
    if init_vocabulary_size < 0:
        raise ValueError("init_vocabulary_size can not be smaller than zero.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be greater than zero.")
    if (embedding_type != "PS") and (embedding_type != "data_parallel"):
        raise TypeError("embedding_type must be PS or data_parallel")


class EmbeddingServiceOut:
    """
    EmbeddingServiceOut.
    """
    def __init__(self, table_id_dict, es_initializer=None, es_counter_filter=None,
                 es_padding_keys=None, es_completion_keys=None):
        self.table_id_dict = table_id_dict
        self.es_initializer = es_initializer
        self.es_counter_filter = es_counter_filter
        self.es_padding_keys = es_padding_keys
        self.es_completion_keys = es_completion_keys


class EmbeddingService:
    r"""
    Currently, ES(EmbeddingService) feature can only create one object which can support model training and inference
    for PS embedding and data_parallel embedding, and provide unified embedding management, storage,
    and computing capabilities for training and inference.
    PS embedding refer to tables that vocab_size more than 100,000, and recommended to store them on the
    Parameter Server (PS). Data_parallel embedding refer to tables that vocab_size less than 100,000, and recommended
    to store them on device.

    .. warning::
        This is an experimental EmbeddingService API that is subject to change.

    .. note::
        This API needs to call 'mindspore.communication.init()' before,
        and it can take effect after the dynamic networking is completed.

    Raises:
        ValueError: If the ESCLUSTER_CONFIG_PATH environment variable is not set during object instantiation.
        ValueError: If the number of each server ParameterServer configured in ESCLUSTER_CONFIG_PATH
            configuration file exceeds four.
        ValueError: If the number of ParameterServer configured in ESCLUSTER_CONFIG_PATH configuration file
            exceeds four.

    Supported Platforms:
        ``Atlas A2 training series products``
    """

    def __init__(self):
        env_dist = os.environ
        es_cluster_config = env_dist.get("ESCLUSTER_CONFIG_PATH")
        if es_cluster_config is None:
            raise ValueError("EsClusterConfig env is null.")
        self._server_ip_to_ps_num = {}
        with open(es_cluster_config, encoding='utf-8') as a:
            es_cluster_config_json = json.load(a)
            self._es_cluster_conf = json.dumps(es_cluster_config_json)
            self._ps_num = int(es_cluster_config_json["psNum"])
            self._ps_ids = []
            self._ps_ids_list = es_cluster_config_json["psCluster"]
            for each_ps in self._ps_ids_list:
                self._server_ip_to_ps_num[each_ps["ctrlPanel"]["ipaddr"]] = 0

            for each_ps in self._ps_ids_list:
                self._ps_ids.append(each_ps["id"])
                ctrl_panel = each_ps["ctrlPanel"]
                self._server_ip_to_ps_num[ctrl_panel["ipaddr"]] += 1

            for each_server_ps_num in self._server_ip_to_ps_num:
                if self._server_ip_to_ps_num[each_server_ps_num] > 4:
                    raise ValueError("PS num of one server can not exceed 4, please check config params.")
                if self._ps_num > 4:
                    raise ValueError("PS num of one server can not exceed 4, please check config params.")

        # storage each ps table's params
        self._table_to_embedding_dim = {}
        self._table_to_max_num = {}
        self._table_to_optimizer = {}
        self._table_to_slot_var_num = {}
        self._table_to_counter_filter = {}
        self._table_id_to_padding_key = {}
        self._table_id_to_completion_key = {}
        self._train_mode = True
        self._train_level = False
        self._optimizer = None
        self._init_table_flag = False

        self._small_table_name_list = []
        self._small_table_variable_list = []
        self._small_table_variable_dim_list = []
        self._ps_table_count = 0
        self._table_name_to_id = {}
        self._table_id_to_name = {}
        self._table_id_to_initializer = {}
        self._table_id_to_steps_to_live = {}

        self._ps_table_id_list = []
        # storage lookup: table_id list, lookup result list, lookup key list
        self._ps_lookup_index = 0
        # storage all inited table names
        self._table_name_has_init = []
        # only storage all inited PS table names
        self._ps_table_name_list = []
        # now only use for adagrad accum
        self._ps_table_id_to_optimizer_params = {}

        # use for data_parallel embedding merge
        self.user_defined_table_infos = []
        self.table_map_policy = None
        self.table_create_infos = []
        self.total_variable_table = []

        # if all data_parallel embedding do not merge
        self._small_table_to_variable = {}
        self._small_table_to_multihot_lens = {}
        self._small_table_name_to_multihot_lens = {}
        self._small_table_name_to_max_vocabulary_size = {}
        self.total_embedding_count = 0
        self._npu_table_to_embedding_dim = {}
        self._need_table_merge = False
        self._small_table_init = False

        # use for counter filter
        self._table_use_counter_filter = {}
        self._use_counter_filter = False
        self._use_evict = False
        self._use_padding_key = False
        self._use_completion_key = False

    def embedding_init(self, name, init_vocabulary_size, embedding_dim, max_feature_count=None,
                       initializer=Uniform(scale=0.01), embedding_type="PS", ev_option=None, multihot_lens=None,
                       optimizer=None, allow_merge=False, optimizer_param=None, mode="train"):
        r"""
        Init for PS embedding and data_parallel embedding.

        Args:
            name (str): The embedding table name.
            init_vocabulary_size (int): The size of embedding table.
            embedding_dim (int): The embedding dim of data in embedding table.
            max_feature_count (int): The count of keys when look up for PS.
            initializer (Initializer): The initialization strategy for the PS embedding, default is ``Uniform``.
            embedding_type (str): The embedding type, configurable parameters ["PS", "data_parallel"],
                ``"PS"`` means initializing PS embedding, ``"data_parallel"`` means initializing data_parallel
                embedding, and default is ``"PS"``.
            ev_option (EmbeddingVariableOption): Properties of the PS embedding,
                is a EmbeddingVariableOption obj which returned by embedding_variable_option function.
                Default is ``None``.
            multihot_lens (int): The param only use when allow_merge is enabled, and not support now.
                Default is ``None``.
            optimizer (str): The type of optimizer in the train mode for PS embedding,
                cannot be shared among each PS embedding, and currently only ``"Adam"``, ``"Ftrl"``, ``"SGD"`` and
                ``"RMSProp"`` are supported, and default is ``None``.
            allow_merge (bool): Whether to enable merge data_parallel embeddings, currently only be False,
                and default is ``False``.
            optimizer_param (float): The "initialize accumulator value" param of optimizer which configured by user,
                representing the init value of moment accumulator, and default is ``None``.
            mode (str): Run mode, configurable parameters ["train", "predict", "export"],
                ``"train"`` means train mode, ``"predict"`` means predict mode, ``"export"`` mean export mode,
                and default is ``"train"``.

        Returns:
            - data_parallel embedding - a dict that contain data_parallel embedding information.
            - PS embedding - EmbeddingServiceOut, the embedding init object that contains PS embedding information,
              which contain five parameters: table_id_dict, es_initializer, es_counter_filter, es_padding_keys,
              es_completion_keys.

              - table_id_dict (dict): key is PS embedding and value is table_id.
              - es_initializer (dict): key is table_id and value is EsInitializer obj
                which means PS embedding parameters.
              - es_counter_filter (dict): key is table_id and value is filter option.
              - es_padding_keys (dict): key is table_id and value is padding key.
              - es_completion_keys (dict): key is table_id amd value is completion key.

        Raises:
            ValueError: If "name", "init_vocabulary_size", "embedding_dim", "max_feature_count" are not set.
            ValueError: If the types of "name", "init_vocabulary_size", "embedding_dim", and "max_feature_count"
                do not match.
            ValueError: If the value of "init_vocabulary_size", "embedding_dim" and "max_feature_count"
                is less than or equal to 0, or the value of "init_vocabulary_size" is bigger than 2147483647.
            ValueError: If the number of PS embedding exceeds 1024.
            ValueError: If the value of "optimizer" not in ["adam", "adagrad", "adamw", "ftrl", "sgd", "rmsprop"].
            TypeError: If the type of "initializer" is not EsInitializer obj or not in
                ["TruncatedNormal", "Uniform", "Constant"].
        """
        check_common_init_params(name=name, init_vocabulary_size=init_vocabulary_size, embedding_dim=embedding_dim,
                                 embedding_type=embedding_type)
        if embedding_type == "data_parallel":
            self._check_and_update_small_init_params(name=name, init_vocabulary_size=init_vocabulary_size,
                                                     embedding_dim=embedding_dim, multihot_lens=multihot_lens,
                                                     allow_merge=allow_merge, initializer=initializer)
            new_small_table_info = dict(
                name=name,
                max_vocabulary_size=init_vocabulary_size,
                embedding_dim=embedding_dim,
                multihot_lens=multihot_lens,
                allow_merge=allow_merge,
                initializer=initializer)
            self.user_defined_table_infos.append(new_small_table_info)
            return new_small_table_info

        table_id = self._check_and_update_ps_init_params(name=name, init_vocabulary_size=init_vocabulary_size,
                                                         max_feature_count=max_feature_count, ev_option=ev_option)
        self._ps_lookup_index = self._ps_table_count
        self._table_to_embedding_dim[table_id] = embedding_dim
        self._table_to_max_num[table_id] = max_feature_count
        # storage the table id for embedding PS table
        self._ps_table_id_list.append(table_id)
        self._ps_table_name_list.append(name)

        if len(self._ps_table_id_list) > 1024:
            raise ValueError("Now only 1024 PS embedding tables can be init.")
        bucket_size = math.ceil(init_vocabulary_size / self._ps_num)
        if optimizer is None:
            self._train_mode = False
            self._table_to_slot_var_num[table_id] = 0
        else:
            self._check_ps_opt_and_initializer(optimizer=optimizer, initializer=initializer, table_id=table_id)
            self._optimizer = optimizer
            self._table_to_optimizer[table_id] = self._optimizer
            self._ps_table_id_to_optimizer_params[table_id] = []
            self._update_optimizer_slot_var_num(table_id=table_id)
            # new train or continue train from a checkpoint
            if initializer is not None:
                self._train_level = True
        filter_mode = self._init_counter_filter(table_id, ev_option)
        self._init_padding_key(table_id, ev_option)
        self._init_completion_key(table_id, ev_option)
        self._init_optimizer_mode_and_params(table_id, optimizer_param)
        es_init_layer = ESInitLayer(self._ps_num, self._ps_ids, self._train_mode, self._train_level, table_id,
                                    bucket_size, embedding_dim, self._table_to_slot_var_num.get(table_id),
                                    self._table_id_to_initializer.get(table_id), filter_mode, optimizer,
                                    self._ps_table_id_to_optimizer_params.get(table_id), max_feature_count, mode)
        es_init_layer()
        return EmbeddingServiceOut(self._table_name_to_id, self._table_id_to_initializer,
                                   self._table_to_counter_filter, self._table_id_to_padding_key,
                                   self._table_id_to_completion_key)

    def padding_param(self, padding_key, mask=True, mask_zero=False):
        r"""
        Init padding key option for each PS embedding.

        Args:
            padding_key (int): The value for padding key, must be a genuine and legal hash key.
            mask (bool): Whether to update padding key. If set to false, it will not be updated. Default is ``True``.
            mask_zero (bool): Whether to update padding key when key is 0. Default is ``False``.

        Returns:
            PaddingParamsOption object.

        Raises:
            TypeError: If the type of "padding_key" is not int.
            TypeError: If the type of "mask" is not bool.
        """
        if not isinstance(padding_key, int):
            raise TypeError("padding_key must be int, please check.")
        if not isinstance(mask, bool):
            raise TypeError("mask must be bool, please check.")
        self._use_padding_key = True
        return PaddingParamsOption(padding_key=padding_key, mask=mask, mask_zero=mask_zero)

    def completion_key(self, completion_key, mask=True):
        r"""
        Init completion key option for each PS embedding.

        Args:
            completion_key (int): The value for completion key.
            mask (bool): Whether to update completion key. If set to false, it will not be updated,
                and default is ``True``.

        Returns:
            CompletionKeyOption object.

        Raises:
            TypeError: If the type of "completion_key" is not int.
            TypeError: If the type of "mask" is not bool.
        """
        if not isinstance(completion_key, int):
            raise TypeError("completion_key must be int, please check.")
        if not isinstance(mask, bool):
            raise TypeError("mask must be bool, please check.")
        self._use_completion_key = True
        completion_key_mask = 1 if mask is True else 0
        return CompletionKeyOption(completion_key=completion_key, mask=completion_key_mask)

    def counter_filter(self, filter_freq, default_key=None, default_value=None):
        r"""
        Init counter filter option for each PS embedding.

        .. note::
            This feature only supports training mode. When user set counter filter option in train mode and then eval,
            the default value can be used for the key that cannot be look up in eval.

        Args:
            filter_freq (int): The frequency threshold value for feature admission.
            default_key (int): The key that number of occurrences does not reach the threshold,
                return value of default key as the corresponding value when look up embedding, and default is ``None``.
            default_value (int/float): The key that number of occurrences does not reach the threshold,
                return default value which length value is embedding dim, and default is ``None``.

        Returns:
            CounterFilter object.

        Raises:
            TypeError: If the type of "filter_freq" is not int.
            ValueError: If the value of "filter_freq" is less than 0.
            ValueError: If the values of "default_key" and "default_value" are None.
            ValueError: If neither of the values of "default_key" and "default_value" are None.
            TypeError: If the value of "default_key" is None and the type of "default_value"
                is neither int nor float.
            TypeError: If the value of "default_value" is None and the type of "default_key" is not int.
        """
        if not isinstance(filter_freq, int):
            raise TypeError("filter_freq must be int, please check.")
        if filter_freq < 0:
            raise ValueError("filter_freq must can not be smaller than 0.")
        if (default_key is None) and (default_value is None):
            raise ValueError("default_key and default_value can not be both None.")
        if (default_key is not None) and (default_value is not None):
            raise ValueError("default_key and default_value can not be both set.")
        if default_key is None and (not isinstance(default_value, (int, float))):
            raise TypeError("When default_value is not None, it must be float or int, please check.")
        if default_value is None and (not isinstance(default_key, int)):
            raise TypeError("When default_key is not None, it must be int, please check.")
        self._use_counter_filter = True
        if default_key is None:
            return CounterFilter(filter_freq=filter_freq, default_key_or_value=0,
                                 default_key=0, default_value=default_value)
        return CounterFilter(filter_freq=filter_freq, default_key_or_value=1,
                             default_key=default_key, default_value=1)

    def evict_option(self, steps_to_live):
        r"""
        Set evict option for each PS embedding.

        Args:
            steps_to_live (int): The steps set for evict key.

        Returns:
            EvictOption object.

        Raises:
            TypeError: If the type of "steps_to_live" is not int.
            ValueError: If the value of "steps_to_live" is not greater than 0.
        """
        if not isinstance(steps_to_live, int):
            raise TypeError("steps_to_live must be int, please check.")
        if steps_to_live <= 0:
            raise ValueError("steps_to_live must must be greater than 0.")
        self._use_evict = True
        return EvictOption(steps_to_live=steps_to_live)

    def embedding_variable_option(self, filter_option=None, padding_option=None, evict_option=None,
                                  completion_option=None, storage_option=None, feature_freezing_option=None,
                                  communication_option=None):
        r"""
        Set variable option for PS embedding.

        Args:
            filter_option (CounterFilter): The option of counter filter. Default is ``None``.
            padding_option (PaddingParamsOption): The option of padding key. Default is ``None``.
            evict_option (EvictOption): The option evict. Default is ``None``.
            completion_option (CompletionKeyOption): The option of completion key. Default is ``None``.
            storage_option (None): Reserved option, currently not supported. Default is ``None``.
            feature_freezing_option (None): Reserved option, currently not supported. Default is ``None``.
            communication_option (None): Reserved option, currently not supported. Default is ``None``.

        Returns:
            EmbeddingVariableOption object, used as the ev_option parameter for embedding_init.

        Raises:
            TypeError: If value of "filter_option" is not None and the type of "filter_option" is not CounterFilter.
            TypeError: If value of "padding_option" is not None and the type of "padding_option" is not
                PaddingParamsOption.
            TypeError: If value of "completion_option" is not None and the type of "completion_option" is not
                CompletionKeyOption.
            TypeError: If value of "evict_option" is not None and the type of "evict_option" is not EvictOption.
        """
        if (filter_option is not None) and (not isinstance(filter_option, CounterFilter)):
            raise TypeError("If padding_option isn't None, it must be CounterFilter type.")
        if filter_option is not None:
            self._use_counter_filter = True
        if (padding_option is not None) and (not isinstance(padding_option, PaddingParamsOption)):
            raise TypeError("If padding_option isn't None, it must be EmbeddingPaddingParamsOption type.")
        if (completion_option is not None) and (not isinstance(completion_option, CompletionKeyOption)):
            raise TypeError("If completion_option isn't None, it must be EmbeddingPaddingCompletionKeyOption type.")
        if (evict_option is not None) and (not isinstance(evict_option, EvictOption)):
            raise TypeError("When evict_option is not None, it must be EvictOption type.")
        return EmbeddingVariableOption(filter_option=filter_option, padding_option=padding_option,
                                       evict_option=evict_option, completion_option=completion_option,
                                       storage_option=storage_option, feature_freezing_option=feature_freezing_option,
                                       communication_option=communication_option)

    def embedding_ckpt_export(self, file_path, trainable_var):
        r"""
        Export the embedding table and optimizer parameters of each PS embedding,
        and export embedding of a data_parallel embedding.

        .. note::
            This function can only be executed by rank 0.
            Need to call embedding_variable_option to set evict_option for each PS embedding before export.

        Args:
            file_path (str): The path to export embedding ckpt, and the last character cannot be ``"/"``.
            trainable_var (list[parameter]): The list of data_parallel embedding parameter.

        Returns:
            The output of EmbeddingComputeVarExport operator and data_parallel embedding export result.
        """
        embedding_dim_list = []
        value_total_len_list = []
        steps_to_live_list = []
        feature_mapping_export_list = []
        if self._small_table_variable_list:
            small_table_var = []
            small_table_var_name = []
            for x in trainable_var:
                if hasattr(x, "feature_name"):
                    if x.feature_name in self._small_table_variable_list:
                        small_table_var.append(x)
                        small_table_var_name.append(x.feature_name)
            index = 0
            for var in small_table_var:
                var_name = small_table_var_name[index]
                embedding_dim = self._small_table_variable_dim_list[index]
                embedding_feature_mapping_export_layer = ESEmbeddingFeatureMappingExport(file_path, False, var,
                                                                                         var_name, embedding_dim)
                feature_mapping_export = embedding_feature_mapping_export_layer()
                index += 1
                feature_mapping_export_list.append(feature_mapping_export)

            if self._ps_table_count == 0:
                return feature_mapping_export_list

        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                        (self._table_to_slot_var_num.get(table_id) + 1) + 2)
            steps_to_live_list.append(self._table_id_to_steps_to_live.get(table_id, 0))
        embedding_ckpt_export_layer = ESEmbeddingCKPTExport(embedding_dim_list, value_total_len_list,
                                                            self._ps_table_name_list, self._ps_table_id_list,
                                                            file_path, steps_to_live_list)
        global_step = Tensor([-1], ms.int64)
        return embedding_ckpt_export_layer(global_step), feature_mapping_export_list

    def embedding_table_export(self, file_path, trainable_var):
        r"""
        Export Embedding table for each PS embedding and data_parallel embedding.

        .. note::
            This function can only be executed by rank 0.

        Args:
            file_path (str): The path to export embedding table, and the last character cannot be ``"/"``.
            trainable_var (list[parameter]): The list of data_parallel embedding parameter.

        Returns:
            The output of EmbeddingTableExport operator and data_parallel embedding export result.
        """
        embedding_dim_list = []
        feature_mapping_export_list = []
        if self._small_table_variable_list:
            small_table_var = []
            small_table_var_name = []
            for x in trainable_var:
                if hasattr(x, "feature_name"):
                    if x.feature_name in self._small_table_variable_list:
                        small_table_var.append(x)
                        small_table_var_name.append(x.feature_name)
            index = 0
            for var in small_table_var:
                var_name = small_table_var_name[index]
                embedding_dim = self._small_table_variable_dim_list[index]
                embedding_feature_mapping_export_layer = ESEmbeddingFeatureMappingExport(file_path, True, var,
                                                                                         var_name, embedding_dim)
                feature_mapping_export = embedding_feature_mapping_export_layer()
                index += 1
                feature_mapping_export_list.append(feature_mapping_export)

            if self._ps_table_count == 0:
                return feature_mapping_export_list
        steps_to_live_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            steps_to_live_list.append(self._table_id_to_steps_to_live.get(table_id, 0))

        embedding_table_export_layer = ESEmbeddingTableExport(embedding_dim_list, embedding_dim_list,
                                                              self._ps_table_name_list, self._ps_table_id_list,
                                                              file_path, steps_to_live_list)
        global_step = Tensor([-1], ms.int64)
        return embedding_table_export_layer(global_step), feature_mapping_export_list

    def incremental_embedding_table_export(self, file_path):
        r"""
        Incremental export embedding table for each PS embedding.

        .. note::
            This function can only be executed by rank 0.

        Args:
            file_path (str): The path to incremental export embedding table, and the last character cannot be ``"/"``.

        Returns:
            The output of EmbeddingTableExport op.
        """
        embedding_dim_list = []
        steps_to_live_list = []
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            steps_to_live_list.append(self._table_id_to_steps_to_live.get(table_id, 0))

        incremental_embedding_table_export_layer = ESIncrementalEmbeddingTableExport(embedding_dim_list,
                                                                                     embedding_dim_list,
                                                                                     self._ps_table_name_list,
                                                                                     self._ps_table_id_list,
                                                                                     file_path, steps_to_live_list)
        global_step = Tensor([-1], ms.int64)
        incremental_embedding_table_export_layer(global_step)

    def embedding_ckpt_import(self, file_path):
        r"""
        Import embedding and ckpt file from file path.

        Args:
            file_path (str): The path to import embedding and ckpt, and the last character cannot be ``"/"``.

        Returns:
            The output of EmbeddingComputeVarImport operator and data_parallel embedding import result.
        """
        embedding_dim_list = []
        value_total_len_list = []
        feature_mapping_import_list = []
        if self._small_table_variable_list:
            index = 0
            for i in range(len(self._small_table_variable_list)):
                small_table_name = self._small_table_variable_list[i]
                small_table_embedding_dim = self._small_table_variable_dim_list[i]
                embedding_feature_mapping_import = ESEmbeddingFeatureMappingImport(file_path, small_table_name,
                                                                                   small_table_embedding_dim, False)
                feature_mapping_insert = embedding_feature_mapping_import()
                index += 1
                feature_mapping_import_list.append(feature_mapping_insert)
            if self._ps_table_count == 0:
                return feature_mapping_import_list
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
            value_total_len_list.append(self._table_to_embedding_dim.get(table_id) *
                                        (self._table_to_slot_var_num.get(table_id) + 1) + 2)

        embedding_ckpt_export_layer = ESEmbeddingCKPTImport(embedding_dim_list, value_total_len_list,
                                                            self._ps_table_name_list, self._ps_table_id_list,
                                                            file_path)
        global_step = Tensor([-1], ms.int64)
        return embedding_ckpt_export_layer(global_step), feature_mapping_import_list

    def embedding_table_import(self, file_path):
        r"""
        Import embedding file from file path.

        Args:
            file_path (str): The path to import embedding table, and the last character cannot be ``"/"``.

        Returns:
            The output of EmbeddingTableImport operator and data_parallel embedding import result.
        """
        embedding_dim_list = []
        feature_mapping_import_list = []
        if self._small_table_variable_list:
            index = 0
            for i in range(len(self._small_table_variable_list)):
                small_table_name = self._small_table_variable_list[i]
                small_table_embedding_dim = self._small_table_variable_dim_list[i]
                embedding_feature_mapping_import = ESEmbeddingFeatureMappingImport(file_path, small_table_name,
                                                                                   small_table_embedding_dim, True)
                feature_mapping_insert = embedding_feature_mapping_import()
                index += 1
                feature_mapping_import_list.append(feature_mapping_insert)
            if self._ps_table_count == 0:
                return feature_mapping_import_list
        for table_id in self._ps_table_id_list:
            embedding_dim_list.append(self._table_to_embedding_dim.get(table_id))
        embedding_table_export_layer = ESEmbeddingTableImport(embedding_dim_list, embedding_dim_list,
                                                              self._ps_table_name_list, self._ps_table_id_list,
                                                              file_path)
        global_step = Tensor([-1], ms.int64)
        return embedding_table_export_layer(global_step), feature_mapping_import_list

    def embedding_evict(self, steps_to_live):
        r"""
        Embedding evict for all PS embedding.

        Args:
            steps_to_live (int): The steps set for evict key.

        Returns:
            The output of ESEmbeddingTableEvict op.

        Raises:
            TypeError: If the type of "steps_to_live" is not int.
            ValueError: If the value of "steps_to_live" is not greater than 0.
        """
        if not isinstance(steps_to_live, int):
            raise TypeError("steps_to_live must be int.")
        if steps_to_live <= 0:
            raise ValueError("steps_to_live must be greater than zero.")

        embedding_table_evict = ESEmbeddingTableEvict(var_handle=self._ps_table_id_list,
                                                      global_step=1,
                                                      steps_to_live=steps_to_live)
        return embedding_table_evict()

    def init_table(self):
        r"""
        Init table for data_parallel embedding.

        Returns:
            A dict of data_parallel embedding parameter, that key is data_parallel embedding name,
            value is data_parallel embedding parameter.
        """
        if not self.user_defined_table_infos:
            raise ValueError("data_parallel embedding has not been created.")
        self.total_embedding_count = 0
        self._create_variable_for_small_table()
        return self._small_table_to_variable

    def _check_and_update_ps_init_params(self, name, init_vocabulary_size, max_feature_count, ev_option):
        """
        Check parameter server parameters and init table id.
        """
        steps_to_live = 0
        if max_feature_count is None:
            raise ValueError("For ps table, max_feature_count can not be None.")
        if (ev_option is not None) and (not isinstance(ev_option, EmbeddingVariableOption)):
            raise TypeError("For ps table, ev_option must be EmbeddingVariableOption type.")
        if (ev_option is not None) and (ev_option.evict_option is not None):
            steps_to_live = ev_option.evict_option.steps_to_live
        if not isinstance(max_feature_count, int):
            raise ValueError("For ps table, max_feature_count must be int.")
        if init_vocabulary_size >= _INT32_MAX_VALUE:
            raise ValueError("init_vocabulary_size exceeds int32 max value.")
        if max_feature_count <= 0:
            raise ValueError("For ps table, max_feature_count must be greater than zero.")
        if name not in self._table_name_has_init:
            table_id = self._ps_table_count
            self._table_name_to_id[name] = table_id
            self._table_id_to_name[table_id] = name
            self._table_id_to_steps_to_live[table_id] = steps_to_live
            self._ps_table_count += 1
            self._table_name_has_init.append(name)
        else:
            raise ValueError("This table has been initialized.")
        return table_id

    def _check_ps_opt_and_initializer(self, optimizer, initializer, table_id):
        """
        Check args of parameter server.
        """
        if optimizer not in ["adam", "adagrad", "adamw", "ftrl", "sgd", "rmsprop"]:
            raise ValueError("optimizer should be one of adam, adagrad, adamw, ftrl, sgd, rmsprop")
        if initializer is not None:
            if isinstance(initializer, EsInitializer):
                self._table_id_to_initializer[table_id] = initializer
            elif isinstance(initializer, TruncatedNormal):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="truncated_normal", mu=initializer.mean,
                                  sigma=initializer.sigma, seed=initializer.seed[0])
            elif isinstance(initializer, Uniform):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="random_uniform",
                                  min_scale=-initializer.scale,
                                  max_scale=initializer.scale, seed=initializer.seed[0])
            elif isinstance(initializer, Constant):
                self._table_id_to_initializer[table_id] = \
                    EsInitializer(initializer_mode="constant", constant_value=initializer.value)
            else:
                raise TypeError("initializer must be EsInitializer or mindspore initializer, and only support"
                                "Uniform, TruncatedNormal and Constant value.")

    def _update_optimizer_slot_var_num(self, table_id):
        """
        Update _table_to_slot_var_num by diff optimizer.
        """
        # adam, adamw, rmsprop include m and v, 2 slots; adagrad include accumulator, 1 slot; sgd include 0 slot
        if self._optimizer == "adagrad":
            self._table_to_slot_var_num[table_id] = 1
        elif self._optimizer == "sgd":
            self._table_to_slot_var_num[table_id] = 0
        else:
            self._table_to_slot_var_num[table_id] = 2

    def _init_counter_filter(self, table_id, ev_option):
        """
        Init counter filter parameters.
        """
        if (ev_option is not None) and (ev_option.filter_option is not None):
            filter_mode = "counter"
            self._table_to_counter_filter[table_id] = ev_option.filter_option
            self._table_use_counter_filter[table_id] = 1
        else:
            filter_mode = "no_filter"
            self._table_use_counter_filter[table_id] = 0
        return filter_mode

    def _init_padding_key(self, table_id, ev_option):
        """
        Init padding key parameters.
        """
        if (ev_option is not None) and (ev_option.padding_option is not None):
            self._table_id_to_padding_key[table_id] = ev_option.padding_option

    def _init_completion_key(self, table_id, ev_option):
        """
        Init completion key parameters.
        """
        if (ev_option is not None) and (ev_option.completion_option is not None):
            self._table_id_to_completion_key[table_id] = ev_option.completion_option

    def _init_optimizer_mode_and_params(self, table_id, optimizer_param):
        """
        Init _ps_table_id_to_optimizer_params by diff optimizer.
        """
        optimizer = self._table_to_optimizer.get(table_id)
        if optimizer is None:
            return
        if optimizer in ["adagrad", "ftrl", "rmsprop"]:
            if optimizer_param is not None:
                self._ps_table_id_to_optimizer_params[table_id].append(optimizer_param)
            else:
                raise ValueError("Optimizer_param: initial_accumulator_value for adagrad and ftrl, ms for rmsprop")
        if optimizer in ["adam", "adamw", "sgd", "ftrl", "rmsprop"]:
            self._ps_table_id_to_optimizer_params[table_id].append(0.)

    def _create_variable_for_small_table(self):
        """
        Create variable for data_parallel embedding.
        """
        if not self._need_table_merge:
            self._create_variable_when_no_merge()

        self._small_table_init = True

    def _create_variable_when_no_merge(self):
        """
        Create variable for data_parallel embedding when not merge data_parallel embedding.
        """
        for user_table_info in self.user_defined_table_infos:
            self._small_table_to_variable[user_table_info['name']] = Parameter(
                ms_initializer(user_table_info['initializer'],
                               shape=[user_table_info['max_vocabulary_size'] + 1, user_table_info['embedding_dim']],
                               dtype=mstype.float32), name=user_table_info['name'])
            self._small_table_to_multihot_lens[self.total_embedding_count] = user_table_info['multihot_lens']
            self._small_table_name_to_max_vocabulary_size[user_table_info['name']] = \
                user_table_info['max_vocabulary_size']
            self._small_table_name_to_multihot_lens[user_table_info['name']] = \
                user_table_info['multihot_lens']
            self._small_table_variable_list.append(user_table_info['name'])
            self._small_table_variable_dim_list.append(user_table_info['embedding_dim'])
            self.total_embedding_count += 1

    def _check_and_update_small_init_params(self, name, init_vocabulary_size, embedding_dim, multihot_lens, allow_merge,
                                            initializer):
        """
        Check and update data_parallel embedding init parameters.
        """
        if name not in self._small_table_name_list:
            self._small_table_name_list.append(name)
        else:
            raise ValueError("This data_parallel embedding has been initialized.")
        if (init_vocabulary_size is None) or (embedding_dim is None) or (multihot_lens is None):
            raise ValueError("max_vocabulary_size or embedding_dim or multihot_lens can not be None.")
        if (not isinstance(init_vocabulary_size, int)) or (not isinstance(embedding_dim, int)) or \
                (not isinstance(multihot_lens, int)) or (not isinstance(allow_merge, bool)):
            raise TypeError("init_vocabulary_size, embedding_dim, multihot_lens must be int,"
                            "allow_merge must be bool.")
        if init_vocabulary_size <= 0 or embedding_dim <= 0 or multihot_lens <= 0:
            raise ValueError("init_vocabulary_size, embedding_dim, multihot_lens must be greater than zero.")
        if initializer is None:
            raise ValueError("Initializer can not be None.")
        if allow_merge:
            raise ValueError("allow_merge do not support now.")
