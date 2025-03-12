# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore.nn.cell import Cell
from mindspore.parallel.shard import Layout
from mindspore.communication.management import get_rank, get_group_size

class AutoParallel(Cell):
    """
    AutoParallel enable auto parallel configuration for neural network cells. This class provides multiple
    parallel strategies to optimize distributed training across Ascend/GPU devices.

    Args:
        network (Cell): Top-level cell or function in the forward network. Defines the core computational graph
                     structure that will be parallelized. Must be a subclass of `nn.Cell` containing the model
                     architecture.

        parallel_mode (str, optional): Specifies the parallelization strategy engine. Available modes: ``"semi_auto"``,
                    ``"sharding_propagation"``, ``"recursive_programming"``. Default: ``"semi_auto"``.

                     - semi_auto: Achieves data and model parallelism by setting parallel strategies.

                     - sharding_propagation:
                       Automatic strategy propagation mode. Infers sharding strategies for non-annotated operators
                       based on configured operator strategies.

                     - recursive_programming:
                       Full automatic parallelization mode. Dynamically generates parallel strategies through
                       recursive program analysis.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, network, parallel_mode="semi_auto"):
        super(AutoParallel, self).__init__(auto_prefix=False)
        self.network = network

        self._parallel_mode = parallel_mode

        self._global_rank = get_rank()
        self._device_num = get_group_size()

        self._init_param_in_compile = True

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
        self._transformer_opt_config = None

    def no_init_parameters_in_compile(self):
        """
        Suppress the parameter initialization process in the compilation procedure.

        .. warning::
        This is an experimental interface, may be changed or canceled in the future;
        """
        self._init_param_in_compile = False

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
        """By default, MindSpore only loads and saves trainable parameters. This API enables the loading and saving of
        non-trainable parameters as well."""
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

        Raises:
            ValueError: If the type of 'config' is str, but it's value is not 'full_batch' or 'data_parallel'.
            TypeError: When 'config' is not str type nor tuple type.
            TypeError: IF 'config' is tuple type, but it's element is not tuple type nor Layout type.
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
        self._dataset_strategy_config = config

    def hsdp(self, shard_size=-1, threshold=64, optimizer_level="level1"):
        r"""
        Set optimizer parallel configs.

        Args:
            shard_size: Set the optimizer weight shard group size if you want to specific the
                        maximum group size across devices when the parallel optimizer is
                        enabled. The numerical range can be (0, device_num]. Default value
                        is -1, which means the optimizer weight shard group size will
                        the data parallel group of each parameter. Default -1.
            threshold: Set the threshold of parallel optimizer. When parallel optimizer is
                       enabled, parameters with size smaller than this threshold will not be
                       sharded across the devices. Parameter size = shape[0] \* ... \*
                       shape[n] \* size(dtype). Non-negative. Unit: KB. Default: 64.
            optimizer_level: optimizer_level configuration is used to specify
                             the splitting level for optimizer sharding. It is important to note that the implementation
                             of optimizer sharding in static graph is inconsistent with dynamic graph like megatron,
                             but the memory optimization effect is the same. When optimizer_level=``level1``,
                             splitting is performed on weights and optimizer state. When optimizer_level=``level2``,
                             splitting is performed on weights, optimizer state, and gradients.
                             When optimizer_level=``level3``, splitting is performed on weights, optimizer state,
                             gradients, additionally, before the backward pass, the weights are further applied with
                             allgather communication to release the memory used by the forward pass allgather.
                             It must be one of [``level1``, ``level2``, ``level3``]. Default: ``level1``.
        """
        self._enable_parallel_optimizer = True
        if not isinstance(shard_size, int) or (shard_size <= 0 and shard_size != -1):
            raise ValueError("shard_size must be a positive integer or -1, but got {}.".format(shard_size))
        self._optimizer_weight_shard_size = shard_size
        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError("threshold must be a positive integer or 0, but got {}.".format(threshold))
        self._parallel_optimizer_threshold = threshold
        if optimizer_level not in ["level1", "level2", "level3"]:
            raise ValueError("Optimizer level should in ['level1', 'level2', 'level3'], but got {}"
                             .format(optimizer_level))
        self._optimizer_level = optimizer_level

    def pipeline(self, stages=1, output_broadcast=False, interleave=False,
                 scheduler="1f1b"):
        """
        Configure the number of pipelin_dages, whether to broadcast the results,
        whether to enable interleaving scheduling, configure type of scheduler when using pipeline parallel.
        Args:
            stages (int): Set the stage information for pipeline parallelism
                This indicates how the devices are individually distributed on the pipeline.
                All devices will be divided into stages of pipine_dags. Default value: 1.
            output_broadcast (bool): When performing pipeline parallel inference,
                whether the result of the last stage is broadcasted to the other stages. Default value: False.
            interleave (bool): Whether to enable interleaving scheduling.
            scheduler(str): The type of scheduler
        Raises:
            TypeError: If the type of 'stages is not int
            ValueError: When stages <= 0
            TypeError: If the type of 'output_broadcast' is not bool
            TypeError: If the type of 'interleave' is not bool
            TypeError: If the type of 'scheduler' is not str
            ValueError: If the type of 'scheduler' is not supported
        """
        if not isinstance(stages, int):
            raise TypeError("For 'AutoParallel.pipeline', the argument 'stages' "
                            "must be int type, but got the type : {}.".format(type(stages)))
        if stages <= 0:
            raise ValueError("For 'AutoParallel.pipeline', the argument 'stages' "
                             "must be larger than zero, but got value: {}.".format(stages))
        if not isinstance(output_broadcast, bool):
            raise TypeError("For 'AutoParallel.pipeline', the argument 'stages' "
                            "must be bool type, but got the type : {}.".format(type(output_broadcast)))
        if not isinstance(interleave, bool):
            raise TypeError("For 'AutoParallel.pipeline', the argument 'stages' "
                            "must be bool type, but got the type : {}.".format(type(interleave)))
        if not isinstance(scheduler, str):
            raise TypeError("For 'AutoParallel.pipeline', the argument 'stages' "
                            "must be str type, but got the type : {}.".format(type(scheduler)))
        if scheduler not in ("1f1b", "gpipe"):
            raise ValueError("For 'AutoParallel.pipeline', the argument "
                             "'scheduler' must be '1f1b' , 'gpipe' , but got the value : {}."
                             .format(scheduler))
        self._pipeline_stages = stages
        self._pipeline_result_broadcast = output_broadcast
        self._pipeline_interleave = interleave
        self._pipeline_scheduler = scheduler

    def comm_fusion(self, config):
        r"""
        Set fusion method for auto parallel.

        Args:
            config (dict): A dict contains the types and configurations for setting the communication fusion. each
                communication fusion config has two keys: "mode" and "config".
                It supports following communication fusion types and configurations:

                - openstate: Whether turn on the communication fusion or not. If `openstate` is ``True`` ,
                  turn on the communication fusion, otherwise, turn off the communication fusion.
                  Default: ``True`` .
                - allreduce: If communication fusion type is `allreduce`. The `mode` contains: `auto`, `size`
                  and `index`. In `auto` mode, AllReduce fusion is configured by gradients size and the default
                  fusion threshold is `64` MB. In 'size' mode, AllReduce fusion is configured by gradients size
                  manually, and the fusion threshold must be larger than `0` MB. In `index` mode, it is same as
                  `all_reduce_fusion_config`.
                - allgather: If communication fusion type is `allgather`. The `mode` contains: `auto`, `size`.
                  In `auto` mode, AllGather fusion is configured by gradients size, and the default fusion
                  threshold is `64` MB. In 'size' mode, AllGather fusion is configured by gradients size
                  manually, and the fusion threshold must be larger than `0` MB.
                - reducescatter: If communication fusion type is `reducescatter`. The `mode` contains: `auto`
                  and `size`. Config is same as `allgather`.

        Raises:
            KeyError: If the type of config is not dict.
        """
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"The parameter '{config}' must be {dict}, but got {type(config)}")
        self._comm_fusion_config = config

    def enable_fp32_communication(self):
        """
        Enable reduce operators (AllReduce, ReduceScatter) are forced to use the fp32 data type for communication
        during communication.
        """
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
        """ Print local norm value for auto parallel. """
        self._dump_local_norm = True

    def dump_local_norm(self, file_path):
        """
        Enable local norm printing with disk storage only (no console output).

        Args:
            file_path (str): The path to save local_norm.

        Raises:
            TypeError: If the type of 'file_path' is not str.
        """
        if not isinstance(file_path, str):
            raise TypeError("the argument 'file_path' must be str, but got the type : {} .".format(type(file_path)))
        self._dump_local_norm = True
        self._dump_local_norm_path = file_path

    def enable_device_local_norm(self):
        """Enable device local norm printing."""
        self._dump_device_local_norm = True

    def enable_gradients_mean(self):
        """ Perform mean operator after allreduce of gradients in parallel mode. """
        self._gradients_mean = True

    def disable_gradient_fp32_sync(self):
        """ Disable convert tensor type from fp16 to fp32 before parameter gradients allreduce. """
        self._gradient_fp32_sync = False

    def disable_loss_repeated_mean(self):
        """
        The mean operator is not executed backwards when the calculation is repeated.
        """
        self._loss_repeated_mean = False

    def transformer_opt(self, file_path):
        r"""
        Check and set speedup config for auto parallel.

        Args:
            file_path(Union[str, None]): The path to the parallel speed up json file, configuration can refer to
            `parallel_speed_up.json
            <https://gitee.com/mindspore/mindspore/blob/master/config/parallel_speed_up.json>`_ .
            If its value is None or '', it does not take effect. Default None.

             - recomputation_communication_overlap (bool): Enable overlap between recompute ops and communication ops
               if True.
               Default: False.
             - grad_matmul_communication_overlap (bool): Enable overlap between dw matmul and
               tensor parallel communication ops if True. Default: False.
             - grad_fa_allgather_overlap (bool): Enable overlap between duplicated allgather by recomputing
               in sequence parallel and flashattentionscoregrad ops if True. Default: False.
             - enable_communication_fusion (bool): Enable communication fusion to optimize the number of communication
               operator tasks if True.
               Default: False.
             - grad_computation_allreduce_overlap (bool): Enable overlap between dx ops and data parallel communication
               ops if True.
               Currently, do not support
               `O2 <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.JitConfig.html>`_
               Default: False.
             - computation_allgather_overlap (bool): Enable overlap between forward ops
               and optimizer parallel allgather communication if True. Currently, do not support
               `O2 <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.JitConfig.html>`_
               Default: False.
             - computation_communication_fusion_level (int): Enable the fusion between compute and communicate.
               Default: ``0``. Note: This function must be used with Ascend Training Solution 24.0.RC2 or later.

               - 0: Disable fusion.

               - 1: Apply fusion to forward nodes.

               - 2: Apply fusion to backward nodes.

               - 3: Apply fusion to all nodes.
             - dataset_broadcast_opt_level (int): Optimize the scenario that the dataset repeated reading. Only
               support O0/O1 jit level. It doesn't work in O2 mode. Default: ``0``.

               - 0: Disable this optimize.

               - 1: Optimize dataset reader between pipeline stage.

               - 2: Optimize dataset reader within pipeline stage.

               - 3: Optimize dataset reader with all scenes.
             - allreduce_and_biasadd_swap (bool): Enable node execution order swap communication operators and add
               operators if ``True``. Only 1-dimension bias node is supported. Default: ``False``.
             - enable_allreduce_slice_to_reducescatter (bool): Enable allreduce optimization. In the scenario where
               the batchmatmul model introduces allreduce in parallel, if the subsequent nodes are stridedslice
               operator with model parallel, allreduce will be optimized as reducescatter according to the identified
               patterns. Typical used in MoE module with groupwise alltoall. Default: ``False``.
             - enable_interleave_split_concat_branch (bool): Enable communication computation parallel optimization
               for branches formed by split and concat operators with ``enable_interleave`` attribute. It is typical
               used in MoE parallel scenario. After splitting the input data, each slice of data is processed by the
               MoE module, and then the branch results are concatenated. When the optimization is enable,
               communication and computation will be executed in parallel between branches. Default: ``False``.
             - enable_interleave_parallel_branch (bool): Enable communication computation parallel optimization
               for parallel branches with ``parallel_branch`` attribute in branches merge node. It is typical
               used in MoE parallel scenario with routed and shared expert. When the optimization is enable,
               communication and computation will be executed in parallel between branches. Default: ``False``.

        Examples:
            >>> net = AutoParallel(net)
            >>> net.transformer_opt("./parallel_speed_up.json")
        """
        # disable pylint too broad Exception
        # pylint: disable=W0212
        from mindspore.context import _context
        ctx = _context()
        ctx._set_speedup_config_path(file_path)
        self._transformer_opt_config = file_path
        ctx.ascend_config['parallel_speed_up_json_path'] = file_path

    def construct(self, *inputs):
        return self.network(*inputs)
