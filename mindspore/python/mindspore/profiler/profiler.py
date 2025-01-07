# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""Profiling api file."""
import os
import json
from typing import Optional, Dict
from sys import getsizeof
from concurrent.futures import ProcessPoolExecutor, as_completed

from mindspore import log as logger
from mindspore.profiler.common.constant import ProfilerStepNameConstant, DeviceTarget
from mindspore.profiler.common.profiler_context import ProfilerContext
from mindspore.profiler.platform.npu_profiler import NPUProfilerAnalysis
from mindspore.profiler.profiler_action_controller import ProfilerActionController
from mindspore.profiler.profiler_interface import ProfilerInterface
from mindspore.profiler.schedule import _default_schedule_fn, ProfilerAction
from mindspore.profiler.common.record_function import RecordFunction
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.profiler_path_manager import ProfilerPathManager


def tensor_board_trace_handler():
    """
    For each step in dynamic graph mode, call this method for online analyse.

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import context, nn, Profiler
        >>> from mindspore.profiler import schedule, tensor_board_trace_handler
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2, 2)
        ...
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator_net():
        ...     for _ in range(2):
        ...         yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)
        >>>
        >>> def train(test_net):
        ...     optimizer = nn.Momentum(test_net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator_net(), ["data", "label"])
        ...     model = ms.train.Model(test_net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        ...
        ...     net = Net()
        ...     STEP_NUM = 15
        ...
        ...     with Profiler(schedule=schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=2),
        ...                   on_trace_ready=tensor_board_trace_handler) as prof:
        ...         for i in range(STEP_NUM):
        ...             train(net)
        ...             prof.step()
    """

    try:
        NPUProfilerAnalysis.online_analyse()
        if ProfilerContext().data_simplification:
            ProfilerPathManager().simplify_data()
    except Exception as e:  # pylint: disable=W0703
        logger.error("Call tensorboard_trace_handler failed. Exception: %s", str(e))


class Profiler:
    r"""
    This class to enable the profiling of MindSpore neural networks.
    MindSpore users can import the mindspore.Profiler, initialize the Profiler object to start profiling,
    and use Profiler.analyse() to stop profiling and analyse the results.
    Users can visualize the results using the `MindStudio Insight
    <https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann>`_ tool.
    Now, Profiler supports AICORE operator, AICPU operator, HostCPU operator, memory,
    correspondence, cluster, etc data analysis.

    Args:
        start_profile (bool, optional): The start_profile parameter controls whether to enable or disable performance
            data collection based on conditions. Default: ``True`` .
        output_path (str, optional): Output data path. Default: ``"./data"`` .
        profiler_level (ProfilerLevel, optional): (Ascend only) The level of profiling.
            Default: ``ProfilerLevel.Level0``.

            - ProfilerLevel.Level0: Leanest level of profiling data collection, collects information about the elapsed
              time of the computational operators on the NPU and communication large operator information.
            - ProfilerLevel.Level1: Collect more CANN layer AscendCL data and AICore performance metrics and
              communication mini operator information based on Level0.
            - ProfilerLevel.Level2: Collect GE and Runtime information in CANN layer on top of Level1
        activities (list, optional): The activities to collect.
            Default: ``[ProfilerActivity.CPU, ProfilerActivity.NPU]``.

            - ProfilerActivity.CPU: Collect MindSpore framework data.
            - ProfilerActivity.NPU: Collect CANN software stack and NPU data.
            - ProfilerActivity.GPU: Collect GPU data.
        schedule (schedule, optional): Sets the action strategy for the capture, defined by the schedule class,
            to be used with the step interface. Default: ``None``.
        on_trace_ready (Callable, optional): Sets the callback function to be executed when the performance data
            is collected. Default: ``None``.
        profile_memory (bool, optional): (Ascend only) Whether to collect tensor memory data, collect when ``True`` .
            When using this parameter, `activities` must set to ``[ProfilerActivity.CPU, ProfilerActivity.NPU]``.
            Collecting operator memory data when the graph compilation level is O2 requires collecting from the
            first step. Default: ``False`` .
        aicore_metrics (AicoreMetrics, optional): (Ascend only) Types of AICORE performance data collected,
            when using this parameter, `activities` must include ``ProfilerActivity.NPU`` , and the value
            must be a member of AicoreMetrics. Default: ``AicoreMetrics.AiCoreNone`` .
            The data items contained in each metric are as follows:

            - AicoreMetrics.AiCoreNone: Does not collect AICORE data.
            - AicoreMetrics.ArithmeticUtilization: ArithmeticUtilization contains mac_fp16/int8_ratio,
              vec_fp32/fp16/int32_ratio, vec_misc_ratio etc.
            - AicoreMetrics.PipeUtilization: PipeUtilization contains vec_ratio, mac_ratio, scalar_ratio,
              mte1/mte2/mte3_ratio, icache_miss_rate etc.
            - AicoreMetrics.Memory: Memory contains ub_read/write_bw, l1_read/write_bw, l2_read/write_bw,
              main_mem_read/write_bw etc.
            - AicoreMetrics.MemoryL0: MemoryL0 contains l0a_read/write_bw, l0b_read/write_bw, l0c_read/write_bw etc.
            - AicoreMetrics.ResourceConflictRatio: ResourceConflictRatio contains vec_bankgroup/bank/resc_cflt_ratio
              etc.
            - AicoreMetrics.MemoryUB: MemoryUB contains ub_read/write_bw_mte, ub_read/write_bw_vector,
              ub\_/write_bw_scalar etc.
            - AicoreMetrics.L2Cache: L2Cache contains write_cache_hit, write_cache_miss_allocate, r0_read_cache_hit,
              r1_read_cache_hit etc. This function only support Atlas A2 training series products.
        with_stack (bool, optional): (Ascend) Whether to collect frame host call stack data on the Python side. This
            data is presented in the form of a flame graph in the timeline. When using this parameter, `activities` must
            include ``ProfilerActivity.CPU``. Default value: ``False`` .
        data_simplification (bool, optional): (Ascend only) Whether to remove FRAMEWORK data and other redundant data.
            If set to True, only the delivery of profiler and the original performance data in the PROF_XXX
            directory are retained to save disk space.
            Default value: ``True`` .
        l2_cache (bool, optional): (Ascend only) Whether to collect l2 cache data, collect when True.
            Default: ``False`` .
        hbm_ddr (bool, optional): (Ascend only) Whether to collect On-Chip Memory/DDR read and write rate data,
            collect when True. Default: ``False`` .
        pcie (bool, optional): (Ascend only) Whether to collect PCIe bandwidth data, collect when True.
            Default: ``False`` .
        data_process (bool, optional): (Ascend/GPU) Whether to collect data to prepare performance data.
            Default value: ``False`` .
        parallel_strategy (bool, optional): (Ascend only) Whether to collect parallel policy performance data.
            Default value: ``False`` .
        sync_enable (bool, optional): (GPU only) Whether the profiler collects operators in a synchronous way.
            Default: ``True`` .

            - True: The synchronous way. Before sending the operator to the GPU, the CPU records the start timestamp.
              Then the operator is returned to the CPU after execution, and the end timestamp is recorded,
              The duration of the operator is the difference between the two timestamps.
            - False: The asynchronous way. The duration of the operator is that of sending from the CPU to the GPU.
              This method can reduce the impact of adding profiler on overall training time.
    Raises:
        RuntimeError: When the version of CANN does not match the version of MindSpore,
            MindSpore cannot parse the generated ascend_job_id directory structure.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> import mindspore.dataset as ds
        >>> from mindspore import Profiler
        >>> from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2,2)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator():
        ...     for i in range(2):
        ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
        >>>
        >>> def train(net):
        ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator, ["data", "label"])
        ...     model = ms.train.Model(net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is GPU, set the device_target to "GPU"
        ...     ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        ...
        ...     # Init Profiler
        ...     # Note that the Profiler should be initialized before model.train
        ...     profiler = Profiler(profiler_level=ProfilerLevel.Level0,
        ...                         activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        ...                         aicore_metrics=AicoreMetrics.AiCoreNone)
        ...
        ...     # Train Model
        ...     net = Net()
        ...     train(net)
        ...
        ...     # Profiler end
        ...     profiler.analyse()
    """
    MAX_META_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self, **kwargs) -> None:
        self._metadata: Dict[str, str] = {}
        self._prof_context: ProfilerContext = ProfilerContext()
        self._prof_context.set_params(**kwargs)
        self._has_started: bool = False
        self.schedule_arg = kwargs.get('schedule')
        if self.schedule_arg is not None:
            self.schedule = self._prof_context.schedule
            self._record_steps: bool = True
            self._schedule_no_use_step = True
        else:
            self.schedule = _default_schedule_fn
            self._record_steps: bool = False
            self._schedule_no_use_step = None
        self._step_rec_fn: Optional[RecordFunction] = None
        self.step_num = 0
        self.current_action: ProfilerAction = self.schedule(self.step_num)
        self.action_controller = ProfilerActionController(ProfilerInterface, self._prof_context.on_trace_ready)
        if self._prof_context.start_profile:
            self.start()

    def start(self) -> None:
        """
        Turn on Profiler data collection. Profiler can be turned on by condition.

        Raises:
            RuntimeError: If the profiler has already started.
            RuntimeError: If the `start_profile` parameter is not set or is set to ``True``.

        Examples:
            >>> from mindspore.train import Callback
            >>> from mindspore import Profiler
            >>> class StopAtStep(Callback):
            ...     def __init__(self, start_step, stop_step):
            ...         super(StopAtStep, self).__init__()
            ...         self.start_step = start_step
            ...         self.stop_step = stop_step
            ...         self.profiler = Profiler(start_profile=False)
            ...
            ...     def step_begin(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         step_num = cb_params.cur_step_num
            ...         if step_num == self.start_step:
            ...             self.profiler.start()
            ...
            ...     def step_end(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         step_num = cb_params.cur_step_num
            ...         if step_num == self.stop_step:
            ...             self.profiler.stop()
            ...
            ...     def end(self, run_context):
            ...         self.profiler.analyse()
        """
        if self._has_started:
            logger.warning("The profiler has already started. Do not turn on again in the open state.")
            return
        self._has_started = True
        self.action_controller.transit_action(ProfilerAction.NONE, self.current_action)
        if self._record_steps:
            self._step_rec_fn = RecordFunction(ProfilerStepNameConstant.PROFILER_STEP + str(self.step_num))
            self._step_rec_fn.start()

    def stop(self) -> None:
        """
        Turn off Profiler data collection. Profiler can be turned off by condition.

        Raises:
            RuntimeError: If the profiler has not started, this function is disabled.

        Examples:
            >>> from mindspore.train import Callback
            >>> from mindspore import Profiler
            >>> class StopAtEpoch(Callback):
            ...     def __init__(self, start_epoch, stop_epoch):
            ...         super(StopAtEpoch, self).__init__()
            ...         self.start_epoch = start_epoch
            ...         self.stop_epoch = stop_epoch
            ...         self.profiler = Profiler(start_profile=False)
            ...
            ...     def epoch_begin(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         epoch_num = cb_params.cur_epoch_num
            ...         if epoch_num == self.start_epoch:
            ...             self.profiler.start()
            ...
            ...     def epoch_end(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         epoch_num = cb_params.cur_epoch_num
            ...         if epoch_num == self.stop_epoch:
            ...             self.profiler.stop()
            ...
            ...     def end(self, run_context):
            ...         self.profiler.analyse()
        """
        if self._schedule_no_use_step:
            logger.warning("The profiler has schedule. Please use step() to collect data.")
            return
        if not self._has_started:
            logger.error("The profiler has not started. Do not turn off again in the closed state.")
            return
        self._has_started = False
        if self._record_steps and self._step_rec_fn:
            self._step_rec_fn.stop()
        if self.schedule_arg:
            self.action_controller.transit_action(self.current_action, None)
        else:
            ProfilerInterface.stop()
        self._dump_metadata()

    def analyse(self, offline_path=None, pretty=False, step_list=None, mode="sync") -> None:
        """
        Collect and analyze training performance data, support calls during and after training. The example shows above.

        Args:
            offline_path (Union[str, None], optional): The data path which need to be analyzed with offline mode.
                Offline mode isused in abnormal exit scenario. This parameter should be set to ``None``
                for online mode. Default: ``None``.
            pretty (bool, optional): Whether to pretty json files. Default: ``False``.
            step_list (list, optional): A list of steps that need to be analyzed, the steps must be
                consecutive integers. Default: ``None``. By default, all steps will be analyzed.
            mode (str, optional): Analysis mode, it must be one of ["sync", "async"]. Default: ``sync``.

                - sync: analyse data in current process, it will block the current process.
                - async: analyse data in subprocess, it will not block the current process. Since the parsing process
                  will take up extra CPU resources, please enable this mode according to the actual resource situation.

        Examples:
            >>> from mindspore.train import Callback
            >>> from mindspore import Profiler
            >>> class StopAtStep(Callback):
            ...     def __init__(self, start_step=1, stop_step=5):
            ...         super(StopAtStep, self).__init__()
            ...         self.start_step = start_step
            ...         self.stop_step = stop_step
            ...         self.profiler = Profiler(start_profile=False)
            ...
            ...     def step_begin(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         step_num = cb_params.cur_step_num
            ...         if step_num == self.start_step:
            ...             self.profiler.start()
            ...
            ...     def step_end(self, run_context):
            ...         cb_params = run_context.original_args()
            ...         step_num = cb_params.cur_step_num
            ...         if step_num == self.stop_step:
            ...             self.profiler.stop()
            ...
            ...     def end(self, run_context):
            ...         self.profiler.analyse(step_list=[2,3,4], mode="sync")
        """
        if self._has_started:
            ProfilerInterface.stop()
            self._has_started = False

        if self.schedule_arg:
            logger.warning("The profiler has schedule. Please use 'on_trace_ready' to analyse data.")
            return

        if offline_path:
            logger.warning("The parameter 'offline_path' for Profiler.analyse() is deprecated, "
                           "please use Profiler.offline_analyse() instead.")

        self._prof_context.pretty = pretty
        self._prof_context.step_list = step_list
        self._prof_context.mode = mode

        ProfilerInterface.finalize()
        ProfilerInterface.analyse()
        ProfilerInterface.clear()

    @classmethod
    def offline_analyse(cls, path: str, pretty=False, step_list=None, data_simplification=True) -> None:
        """
        Analyze training performance data offline, which is invoked after performance data collection is completed.

        Args:
            path (str): The profiling data path which need to be analyzed offline.
                There needs to be a profiler directory in this path.
            pretty (bool, optional): Whether to pretty json files. Default: ``False``.
            step_list (list, optional): A list of steps that need to be analyzed, the steps must be
                consecutive integers. Default: ``None``. By default, all steps will be analyzed.
            data_simplification (bool, optional): Whether to enable data simplification. Default: ``True``.

        Examples:
            >>> from mindspore import Profiler
            >>> Profiler.offline_analyse("./profiling_path")
        """
        real_path = PathManager.get_real_path(path)
        PathManager.check_input_directory_path(real_path)
        ascend_ms_path_list = PathManager.get_ascend_ms_path_list(real_path)

        if not ascend_ms_path_list:
            msg = (f"Invalid path: {real_path}. Expected a *_ascend_ms_* directory "
                   "or a parent directory of multiple *_ascend_ms_*")
            logger.error(msg)
            return

        worker_number = min(os.cpu_count() // 2, len(ascend_ms_path_list))
        with ProcessPoolExecutor(max_workers=worker_number) as executor:
            futures = [
                executor.submit(
                    NPUProfilerAnalysis.offline_analyse,
                    ascend_ms_path,
                    pretty,
                    step_list,
                    data_simplification
                ) for ascend_ms_path in ascend_ms_path_list
            ]
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e: # pylint: disable=W0703
                    logger.error("offline analysis failed: %s", str(e))

    def step(self) -> None:
        """
        Used for Ascend, distinguish step collection and parsing performance data through schedule and on_trace_ready.

        Raises:
            RuntimeError: If the `start_profile` parameter is not set or the Profiler is not started.
            RuntimeError: If the `schedule` parameter is not set.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> import mindspore.dataset as ds
            >>> from mindspore import context, nn, Profiler
            >>> from mindspore.profiler import schedule, tensor_board_trace_handler
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2, 2)
            ...
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator_net():
            ...     for _ in range(2):
            ...         yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)
            >>>
            >>> def train(test_net):
            ...     optimizer = nn.Momentum(test_net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator_net(), ["data", "label"])
            ...     model = ms.train.Model(test_net, loss, optimizer)
            ...     model.train(1, data)
            >>>
            >>> if __name__ == '__main__':
            ...     context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
            ...
            ...     net = Net()
            ...     STEP_NUM = 15
            ...
            ...     with Profiler(schedule=schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=2),
            ...                   on_trace_ready=tensor_board_trace_handler) as prof:
            ...         for i in range(STEP_NUM):
            ...             train(net)
            ...             prof.step()
        """
        if self.schedule_arg is None:
            logger.error("With no schedule in the Profiler, step takes no effect!")
            return
        if not self._has_started:
            logger.error("Profiler is stopped, step takes no effect!")
            return
        if self._step_rec_fn:
            self._step_rec_fn.stop()
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        self.action_controller.transit_action(prev_action, self.current_action)
        self._step_rec_fn = RecordFunction(ProfilerStepNameConstant.PROFILER_STEP + str(self.step_num))
        self._step_rec_fn.start()
        self._schedule_no_use_step = False

    def add_metadata(self, key: str, value: str):
        """
        Report custom metadata key-value pair data.

        Args:
            key (str): The key to the metadata.
            value (str): The value to the metadata.

        Examples:
            >>> from mindspore import Profiler
            >>> # Profiler init.
            >>> profiler = Profiler()
            >>> # Call Profiler add_metadata
            >>> profiler.add_metadata("test_key", "test_value")
            >>> # Profiler end
            >>> profiler.analyse()
        """
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning("The key and value of metadata must be string. Skip this metadata.")
            return

        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self._metadata) + add_size < self.MAX_META_SIZE:
            if key in self._metadata:
                logger.warning(f"{key} is already saved as metadata, override it.")
            self._metadata[key] = value
        else:
            logger.warning("Too many metadata added. Skip this metadata")

    def add_metadata_json(self, key: str, value: str):
        """
        Report custom metadata key-value pair data with the value as a JSON string data.

        Args:
            key (str): The key to the metadata.
            value (str): The json str format value to the metadata.

        Examples:
            >>> import json
            >>> from mindspore import Profiler
            >>> # Profiler init.
            >>> profiler = Profiler()
            >>> # Call Profiler add_metadata_json
            >>> profiler.add_metadata_json("test_key", json.dumps({"key1": 1, "key2": 2}))
            >>> # Profiler end, metadata will be saved in profiler_metadata.json
            >>> profiler.analyse()
        """
        if not isinstance(key, str) or not isinstance(value, str):
            logger.warning("The key and value of metadata must be string. Skip this metadata.")
            return

        add_size = getsizeof(key) + getsizeof(value)
        if getsizeof(self._metadata) + add_size < self.MAX_META_SIZE:
            try:
                if key in self._metadata:
                    logger.warning(f"{key} is already saved as metadata, override it.")
                self._metadata[key] = json.loads(value)
            except ValueError:
                logger.warning("The metadata value must be json format string. Skip this metadata")
        else:
            logger.warning("Too many metadata added. Skip this metadata")

    def op_analyse(self, op_name, device_id=None):
        """
        Profiler users can use this interface to obtain operator performance data.

        Args:
            op_name (str or list): The primitive operator name to query.
            device_id (int, optional): ID of the target device. This parameter is optional during network training or
                inference, and users can use device_id parameter to specify which card operator performance data to
                parse. If this interface is used for offline data parsing, the default value is ``None`` .

        Raises:
            TypeError: If the `op_name` parameter type is incorrect.
            TypeError: If the `device_id` parameter type is incorrect.
            RuntimeError: If MindSpore runs on Ascend, this interface cannot be used.

        Supported Platforms:
            ``GPU`` ``CPU``

        Examples:
            >>> from mindspore import Profiler
            >>> from mindspore import nn
            >>> from mindspore import Model
            >>> # Profiler init.
            >>> profiler = Profiler()
            >>> # Train Model or eval Model, taking LeNet5 as an example.
            >>> # Refer to https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            >>> # Create the dataset taking MNIST as an example.
            >>> # Refer to https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
            >>> dataloader = create_dataset()
            >>> model = Model(net, loss, optimizer)
            >>> model.train(5, dataloader, dataset_sink_mode=False)
            >>>
            >>> # Profiler end
            >>> profiler.analyse()
            >>>
            >>> profiler.op_analyse(op_name=["BiasAdd", "Conv2D"])
        """
        if self._prof_context.device_target == DeviceTarget.NPU.value:
            raise RuntimeError("The Interface 'Profiler.op_analyse()' is not supported on Ascend currently.")

        if device_id and not isinstance(device_id, int):
            raise TypeError(f"For 'Profiler.op_analyse()', the parameter device_id must be int, "
                            f"but got type {type(device_id)}")

        if not isinstance(op_name, str) and not isinstance(op_name, list):
            raise TypeError(f"For 'Profiler.op_analyse()', the parameter op_name must be str or list, "
                            f"but got type {type(op_name)}")
        if not op_name:
            raise TypeError(f"For 'Profiler.op_analyse()', the parameter op_name cannot be "", '' or [].")

        from mindspore.profiler.parser.framework_parser import GpuFrameWorkParser
        dev_id = self._prof_context.device_id if device_id is None else device_id
        parser = GpuFrameWorkParser(self._prof_context.framework_path, dev_id, op_name)
        op_info = parser.parse()
        return op_info

    def _dump_metadata(self):
        """Dump metadata to file."""
        if not self._metadata:
            return
        save_path = os.path.join(self._prof_context.ascend_ms_dir, "profiler_metadata.json")
        FileManager.create_json_file(save_path, self._metadata, indent=4)
        self._metadata.clear()

    def __enter__(self) -> 'Profiler':
        if not self._has_started:
            self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._has_started:
            self.stop()

    def __del__(self):
        if self._has_started:
            self.stop()
            logger.warning("Profiler is stopped at the end of the program.")
