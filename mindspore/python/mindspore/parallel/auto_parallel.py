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
"""Cell of auto parallel"""
import os
from mindspore import context
from mindspore.nn.cell import Cell
from mindspore.parallel.shard import Layout
from mindspore.communication.management import get_rank, get_group_size

class AutoParallel(Cell):
    """
    AutoParallel Cell
    """
    def __init__(self, network, parallel_mode="semi_auto"):
        super(AutoParallel, self).__init__(auto_prefix=False)
        self.network = network

        self._parallel_mode = parallel_mode

        self._global_rank = get_rank()
        self._device_num = get_group_size()

        self._load_strategy_file_path = ""
        self._save_strategy_file_path = ""
        self._only_trainable_params = True

        self._load_operator_strategy_file = ""
        self._save_operator_strategy_file = ""

        self._dataset_strategy_config = "data_parallel"
        self._full_batch = False

        self._enable_parallel_optimizer = False
        self._optimizer_weight_shard_size = -1
        self._parallel_optimizer_threshold = 64
        self._gradient_accumulation_shard = False

        self._pipeline_stages = 1
        self._pipeline_result_broadcast = False
        self._pipeline_interleave = False
        self._pipeline_scheduler = "1f1b"

        self._comm_fusion_config = dict()

        self._force_fp32_communication = False
        self._enable_alltoall = False
        self._parameter_broadcast = False
        self._group_ckpt_save_file = ""

        self._dump_local_norm = False
        self._dump_local_norm_path = ""
        self._dump_device_local_norm = False

        self._gradients_mean = False
        self._gradient_fp32_sync = True
        self._loss_repeated_mean = True

        self._memory_offload_config = dict()
        self._transformer_opt_config = ""

    def load_param_strategy_file(self, file_path):
        """
        Set the path to load parameter strategy checkpoint file.

        Args:
            file_path (str): The path to load parameter strategy checkpoint.

        Raises:
            TypeError: If the type of 'file_path' is not str
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        self._load_strategy_file_path = file_path

    def save_param_strategy_file(self, file_path):
        """
        Set the path to save parameter strategy checkpoint file.

        Args:
            file_path (str): The path to save parameter strategy checkpoint.

        Raises:
            TypeError: If the type of 'file_path' is not str
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        self._save_strategy_file_path = file_path

    def disable_strategy_file_only_for_trainable_params(self):
        """By default, MindSpore only loads and saves trainable parameters. This API enables the loading and saving of non-trainable parameters as well."""
        self._only_trainable_params = False

    def load_operator_strategy_file(self, file_path):
        """
        Set the path to load strategy json when using sharding propagation.

        .. warning::
        This is an experimental interface, may be changed or canceled in the future;
        This interface currently doesn't support loading strategies using layout.

        Note:
            - It only works when `parallel_mode=sharding_propagation`.
            - When performing distributed training, users can first save the strategy using dryrun on a single device
            and then load strategy to perform distributed training.

        Args:
            file_path (str): Path to load parallel strategy json, must be an absolute path.

        Raises:
            TypeError: If the type of 'file_path' is not str
            KeyError: When 'file_path' is not an absolute path.
            KeyError: When 'file_path' does not end in ``".json"`` .
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        if not os.path.isabs(file_path):
            raise KeyError("the argument 'file_path' must be an absolute path.")
        _, file_type = os.path.splitext(file_path)
        if file_type != ".json":
            raise KeyError("File type must be .json")
        self._load_operator_strategy_file = file_path

    def save_operator_strategy_file(self, file_path):
        """
        Set the path to save strategy json when using sharding propagation.

        .. warning::
        This is an experimental interface, may be changed or canceled in the future;
        This interface currently doesn't support saving strategies using layout.

        Note:
            - It only works when `parallel_mode=sharding_propagation`.
            - When performing distributed training, users can first save the strategy using dryrun on a single device
            and then load strategy to perform distributed training.

        Args:
            file_path (str): Path to save parallel strategy json, must be an absolute path.

        Raises:
            TypeError: If the type of 'file_path' is not str
            KeyError: When 'file_path' is not an absolute path.
            KeyError: When 'file_path' does not end in ``".json"`` .
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        if not os.path.isabs(file_path):
            raise KeyError("the argument 'file_path' must be an absolute path.")
        _, file_type = os.path.splitext(file_path)
        if file_type != ".json":
            raise KeyError("File type must be .json")
        self._save_operator_strategy_file = file_path

    def dataset_strategy(self, config):
        """
        Set dataset sharding strategy.

        Args:
            config Union[str, tuple(tuple), tuple(Layout)]: The dataset sharding strategy.
        """
        if config is None:
            raise ValueError("dataset_strategy is none in config!")

        if isinstance(config, str):
            if config not in ("full_batch", "data_parallel"):
                raise ValueError("For 'AutoParallel.dataset_strategy', the argument "
                                 "'config' must be 'full_batch' or 'data_parallel', but got the value : {}."
                                 .format(config))
            self._full_batch = (config == "full_batch")
            self._dataset_strategy_config = config
            return
        if not isinstance(config, tuple):
            raise TypeError("For 'AutoParallel.dataset_strategy', the argument 'config' "
                            "must be str or tuple type, but got the type : {}.".format(type(config)))
        for ele in config:
            if isinstance(ele, tuple):
                for dim in ele:
                    if not isinstance(dim, int):
                        raise TypeError("For 'AutoParallel.dataset_strategy', the element of argument "
                                        "'config' must be int type, but got the type : {} .".format(type(dim)))
            elif isinstance(ele, Layout):
                pass
            else:
                raise TypeError("For 'AutoParallel.dataset_strategy', the element of argument "
                                "'config' must be tuple or Layout, but got the type : {} .".format(type(ele)))
        if context.get_context('mode') == context.PYNATIVE_MODE:
            raise ValueError("In PyNative mode, the setting value of 'config' must be either 'full_batch' "
                             f"or 'data_parallel', but got {config}.")
        self._dataset_strategy_config = config

    def hsdp(self, shard_size=-1, threshold=64, optimizer_level="level1"):
        self._enable_parallel_optimizer = True
        self._optimizer_weight_shard_size = shard_size
        self._parallel_optimizer_threshold = threshold
        self._optimizer_level = optimizer_level

    def pipeline(self, pipeline_stages=1, pipeline_result_broadcast=False, pipeline_interleave=False,
                 pipeline_scheduler="1f1b"):
        self._pipeline_stages = pipeline_stages
        self._pipeline_result_broadcast = pipeline_result_broadcast
        self._pipeline_interleave = pipeline_interleave
        self._pipeline_scheduler = pipeline_scheduler

    def comm_fusion(self, config):
        self._comm_fusion_config = config

    def enable_fp32_communication(self):
        """Enable fp32 communication."""
        self._force_fp32_communication = True

    def set_group_ckpt_save_file(self, file_path):
        """
        Set the save path of the communication group.

        Args:
            file_path (str): The path to save parallel group checkpoint.

        Raises:
            TypeError: If the type of 'file_path' is not str
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        self._group_ckpt_save_file = file_path

    def print_local_norm(self):
        """Enable local norm printing with console output only (no disk storage)."""
        self._dump_local_norm = True

    def dump_local_norm(self, file_path):
        """Enable local norm printing with disk storage only (no console output)."""
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        self._dump_local_norm = True
        self._dump_local_norm_path = file_path

    def enable_device_local_norm(self):
        """Enable device local norm printing."""
        self._dump_device_local_norm = True

    def enable_gradients_mean(self):
        """Enable mean operation following the gradients allreduce."""
        self._gradients_mean = True

    def disable_gradient_fp32_sync(self):
        """Disable FP32 precision for gradient communication."""
        self._gradient_fp32_sync = False

    def disable_loss_repeated_mean(self):
        """When the loss is computed repeatedly across multiple cards, do not divide the backward gradients by the number of repetitions"""
        self._loss_repeated_mean = False

    def transformer_opt(self, file_path):
        self._transformer_opt_config = file_path

    def auto_memory_offload(self, config):
        self._memory_offload_config = config

    def construct(self, *inputs):
        return self.network(*inputs)
