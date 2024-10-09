# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""Dynamic Profile Monitor"""
import os
import sys
import time
import json
import atexit
import struct
import random
import multiprocessing

from mindspore import log as logger
from mindspore.train import Callback
from mindspore.profiler import Profiler
from mindspore.profiler import ProfilerLevel
from mindspore.communication import get_rank
from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager
from mindspore.profiler.parser.ascend_analysis.path_manager import PathManager


def get_real_rank():
    """get rank id"""
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


def print_msg(msg):
    """print msg"""
    print("[Dynamic Profiler] " + msg, flush=True)


class DynamicProfilerArgs:
    """
    Data class for dynamic profile config.
    """
    FMT = "iiiiii?????"
    SIZE = struct.calcsize(FMT)

    def __init__(self,
                 start_step: int = -1,
                 stop_step: int = -1,
                 aicore_metrics: int = -1,
                 profiler_level: int = -1,
                 profile_framework: int = -1,
                 analyse_mode: int = -1,
                 profile_communication: bool = False,
                 parallel_strategy: bool = False,
                 with_stack: bool = False,
                 data_simplification: bool = True,
                 is_valid: bool = False,
                 **kwargs):
        self._start_step = start_step
        self._stop_step = stop_step
        self._aicore_metrics = aicore_metrics
        self._profiler_level = profiler_level
        self._profile_framework = profile_framework
        self._analyse_mode = analyse_mode
        self._profile_communication = profile_communication
        self._parallel_strategy = parallel_strategy
        self._with_stack = with_stack
        self._data_simplification = data_simplification
        self._is_valid = is_valid
        self._check_params_type()

    def _check_params_type(self):
        """ check params type."""
        if not isinstance(self._start_step, int):
            logger.warning("start_step should be int type, start_step will be reset to -1.")
            self._start_step = -1

        if not isinstance(self._stop_step, int):
            logger.warning("stop_step should be int type, stop_step will be reset to -1.")
            self._stop_step = -1

        if not isinstance(self._aicore_metrics, int):
            logger.warning("aicore_metrics should be int type, aicore_metrics will be reset to -1.")
            self._aicore_metrics = -1

        if not isinstance(self._profiler_level, int):
            logger.warning("profiler_level should be int type, profiler_level will be reset to -1.")
            self._profiler_level = -1

        if not isinstance(self._profile_framework, int):
            logger.warning("profile_framework should be int type, profile_framework will be reset to -1.")
            self._profile_framework = -1

        if not isinstance(self._analyse_mode, int):
            logger.warning("analyse_mode should be int type, analyse_mode will be reset to -1.")
            self._analyse_mode = -1

        if not isinstance(self._profile_communication, bool):
            logger.warning("profile_communication should be bool type, profile_communication will be reset to False.")
            self._profile_communication = False

        if not isinstance(self._parallel_strategy, bool):
            logger.warning("parallel_strategy should be bool type, parallel_strategy will be reset to False.")
            self._parallel_strategy = False

        if not isinstance(self._with_stack, bool):
            logger.warning("with_stack should be bool type, with_stack will be reset to False.")
            self._with_stack = False

        if not isinstance(self._data_simplification, bool):
            logger.warning("data_simplification should be bool type, data_simplification will be reset to True.")
            self._data_simplification = True

        if not isinstance(self._is_valid, bool):
            logger.warning("is_valid should be bool type, is_valid will be reset to False.")
            self._is_valid = False

    @property
    def start_step(self):
        """ get start step value."""
        return self._start_step

    @property
    def stop_step(self):
        """ get stop step value."""
        return self._stop_step

    @property
    def is_valid(self):
        """ get json valid value."""
        return self._is_valid

    @is_valid.setter
    def is_valid(self, value):
        """ set json valid value."""
        self._is_valid = value

    @property
    def analyse_mode(self):
        """ get analyse mode value."""
        return self._convert_analyse_mode(self._analyse_mode)

    @property
    def vars(self):
        """ get all values in DynamicProfilerArgs."""
        not_supported_args = ['_is_valid']
        res = {}
        for key, value in self.__dict__.items():
            if key not in not_supported_args:
                res[key.replace('_', '', 1)] = value
        return res

    @property
    def args(self):
        """ get all args in DynamicProfilerArgs."""
        self._profiler_level = self._convert_profiler_level(self._profiler_level)
        self._profile_framework = self._convert_profile_framework(self._profile_framework)
        not_supported_args = ['_start_step', '_stop_step', '_analyse_mode', '_is_valid']
        res = {}
        for key, value in self.__dict__.items():
            if key not in not_supported_args:
                res[key.replace('_', '', 1)] = value
        return res

    @classmethod
    def from_bytes(cls, byte_data):
        """ unpack bytes to DynamicProfilerArgs."""
        unpacked = struct.unpack(cls.FMT, byte_data)
        return cls(*unpacked)

    def to_bytes(self):
        """ pack DynamicProfilerArgs to bytes."""
        instance_vars = tuple(self.__dict__.values())
        if len(instance_vars) != len(self.FMT):
            raise ValueError("Number of variables does not match format string.")
        return struct.pack(DynamicProfilerArgs.FMT, *instance_vars)

    def _convert_analyse_mode(self, analyse_mode: int) -> str:
        """ convert analyse_mode to real args in Profiler."""
        if analyse_mode == 0:
            return 'sync'
        if analyse_mode == 1:
            return 'async'
        return None

    def _convert_profiler_level(self, profiler_level: int) -> ProfilerLevel:
        """ convert profiler_level to real args in Profiler."""
        if profiler_level == 0:
            return ProfilerLevel.Level0
        if profiler_level == 1:
            return ProfilerLevel.Level1
        if profiler_level == 2:
            return ProfilerLevel.Level2
        return None

    def _convert_profile_framework(self, profile_framework: int) -> str:
        """ convert profile_framework to real args in Profiler."""
        if profile_framework == 0:
            return "time"
        if profile_framework == 1:
            return "all"
        return None


class DynamicProfilerMonitorBase(Callback):
    """
    Dynamic profile callback base class implementing the dynamic profile functionality.
    """

    def __init__(self, cfg_path, output_path=None, poll_interval=2, **kwargs):
        self._cfg_path = cfg_path
        self._cfg_json_path = os.path.join(self._cfg_path, "profiler_config.json")
        self._cfg_json_path = os.path.realpath(self._cfg_json_path)
        self._output_path = "dyn_profile_data" if output_path is None else output_path
        self._poll_interval = poll_interval
        if not isinstance(self._poll_interval, int):
            logger.error("Poll interval must be an integer, reset to 2.")
            self._poll_interval = 2

        if self._poll_interval < 1:
            logger.error("Poll interval must be greater than 1, reset to 2.")
            self._poll_interval = 2

        self._kwargs = kwargs
        self._shm_name = time.strftime("DynamicProfileShm%Y%m%d%H", time.localtime())
        self._rank_id = get_real_rank()
        self._shared_loop_flag = multiprocessing.Value('b', True)
        self._shm = None
        self._process = None
        self._profiler = None
        self._last_start_step = None
        self._last_stop_step = None
        self._is_create_process = None
        self._is_started = False

        self._init_cfg_json()
        self._create_shm()
        self._create_process()
        atexit.register(self._clean_resource)

    def step_begin(self, run_context):
        """
        Start profile at the begin of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        prof_args = self._get_prof_args()

        if not prof_args.is_valid:
            logger.error("Dynamic profile json is not valid, please check the json file.")
            return

        if prof_args.start_step == -1 or prof_args.start_step == self._last_start_step:
            return

        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        start_step, stop_step = self._check_step(prof_args.start_step, prof_args.stop_step, step_num)

        # Prevent repeated calls of the start function within a complete interval
        if step_num == start_step:
            if self._is_started:
                logger.error("Dynamic profile is already started at step %d, "
                             "please wait the first profile finished at step %d.",
                             self._last_start_step, self._last_stop_step)
                return

            if self._profiler is None:
                prof_path = os.path.join(self._output_path, f"rank{self._rank_id}_start{start_step}_stop{stop_step}")
                PathManager.check_input_directory_path(prof_path)
                self._profiler = Profiler(output_path=prof_path, start_profile=False, **prof_args.args)
                print_msg(f"Rank {self._rank_id} create output path {prof_path}")

            self._profiler.start()
            self._is_started = True
            self._last_start_step = start_step
            self._last_stop_step = stop_step
            print_msg(f"Rank {self._rank_id} Dynamic profiler start at step {start_step}, "
                      f"will stop at step {stop_step}")

    def step_end(self, run_context):
        """
        Stop profile at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        prof_args = self._get_prof_args()

        if not prof_args.is_valid:
            logger.error("Dynamic profile json is not valid, please check the json file.")
            return

        if prof_args.stop_step == -1:
            return

        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num

        if step_num == self._last_stop_step and self._is_started:
            if self._profiler:
                self._profiler.stop()
                if prof_args.analyse_mode:
                    self._profiler.analyse(mode=prof_args.analyse_mode)
                else:
                    self._profiler._ascend_profiler.finalize()
                self._profiler = None
                self._is_started = False
                print_msg(f"Rank {self._rank_id} Dynamic profiler stop at step {step_num}")

    def on_train_end(self, run_context):
        """
        Callback on trian end

        Args:
            run_context (RunContext): Context of the train running.
        """
        self._clean_resource()

    def _get_prof_args(self):
        """ Get prof_args """
        logger.error("Dynamic profiler _get_prof_args is not implemented")
        return DynamicProfilerArgs()

    def _clean_resource(self):
        """Clean resource"""
        logger.error("Dynamic profiler _clean_resource is not implemented")

    def _check_step(self, start_step, stop_step, step_num):
        """Check step valid"""
        if start_step <= 0 or stop_step <= 0:
            return -1, -1

        if start_step > stop_step:
            logger.error("start_step must be less than stop_step, "
                         "but get start_step = %d, stop_step = %d", start_step, stop_step)
            return -1, -1

        if start_step < step_num and start_step != self._last_start_step:
            logger.error("start_step must be greater than step_num, "
                         "but get start_step = %d, stop_step = %d, step_num = %d", start_step, stop_step, step_num)
            return -1, -1

        if stop_step < step_num and stop_step != self._last_stop_step:
            logger.error("stop_step must be greater than step_num, "
                         "but get start_step = %d, stop_step = %d, step_num = %d", start_step, stop_step, step_num)
            return -1, -1

        return start_step, stop_step

    def _init_cfg_json(self):
        """Init config json file"""
        if self._rank_id == 0:
            if not os.path.exists(self._cfg_json_path):
                logger.warning("cfg_path is not exist, create default cfg json")
                FileManager.create_json_file(self._cfg_path, DynamicProfilerArgs().vars,
                                             "profiler_config.json", indent=4)
        else:
            logger.info("rank_id is not 0, skip init cfg json")
        print_msg(f"Init config json file: {self._cfg_json_path}")

    def _create_shm(self):
        """Create a json monitor process based on whether the SharedMemory is successfully created"""
        logger.error("Dynamic profiler _create_shm is not implemented")

    def _create_process(self):
        """Create json monitor process, one process will be created at one worker"""
        if self._is_create_process:
            # daemon need to be set to True, otherwise the process will not be killed when the main process exits.
            self._process = multiprocessing.Process(target=worker_func, daemon=True,
                                                    args=(self._shared_loop_flag, self._poll_interval,
                                                          self._shm, self._cfg_json_path))
            self._process.start()
            logger.info("Config monitor process has been created by rank %d.", self._rank_id)
        else:
            self._process = None
            logger.info("Rank %d no need to create process.", self._rank_id)


if sys.version_info >= (3, 8):
    def write_bytes(shm, byte_data):
        """Write bytes to shared memory"""
        shm.buf[:DynamicProfilerArgs.SIZE] = byte_data
else:
    def write_bytes(shm, byte_data):
        """Write bytes to shared memory"""
        shm.seek(0)
        shm.write(byte_data)


def worker_func(loop_flag, poll_interval, shm, cfg_path):
    """ Json monitor process worker function python version >= 3.8"""
    last_file_t = None
    while loop_flag.value:
        if os.path.exists(cfg_path):
            file_t = os.path.getmtime(cfg_path)
            if not last_file_t or last_file_t != file_t:
                last_file_t = file_t

                try:
                    with open(cfg_path, 'r') as f:
                        data = json.load(f)

                    # convert json to DynamicProfilerArgs
                    prof_args = DynamicProfilerArgs(**data)
                    prof_args.is_valid = True
                    logger.info("Dynamic profiler process load json success")
                except json.JSONDecodeError as e:
                    prof_args = DynamicProfilerArgs()
                    prof_args.is_valid = False
                    logger.error("Dynamic profiler process load json failed: %s", e)
                byte_data = prof_args.to_bytes()
                write_bytes(shm, byte_data)
        else:
            logger.error("Dynamic profiler cfg json not exists")
        time.sleep(poll_interval)
    logger.info("Dynamic profiler process done")


if sys.version_info >= (3, 8):
    from multiprocessing import shared_memory
    from unittest.mock import patch

    class DynamicProfilerMonitor(DynamicProfilerMonitorBase):
        r"""
        This class to enable the dynamic profile monitoring of MindSpore neural networks.

        Args:
            cfg_path (str): Dynamic profile json config file directory. The requirement is a shared path
                that can be accessed by all nodes.
            output_path (str, optional): Output data path. Default: ``"./dyn_profile_data"`` .
            poll_interval (int, optional): The polling period of the monitoring process, in seconds.
                Default value: ``2``.

        Raises:
            RuntimeError: When create shared memory times exceeds max times.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore.profiler import DynamicProfilerMonitor
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
            ...     dynprof_cb = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dyn_prof_data")
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # register DynamicProfilerMonitor to model.train()
            ...     model.train(10, data, callbacks=[dynprof_cb])
        """

        def __init__(self, cfg_path, output_path="./dyn_profile_data", poll_interval=2, **kwargs):
            if not isinstance(cfg_path, str):
                raise TypeError("The cfg_path must be a string.")
            if not isinstance(output_path, str):
                logger.warning(f"The output_path must be a string, "
                               f"but got type {type(output_path)}, it will be set to './dyn_profile_data'.")
                output_path = "./dyn_profile_data"
            super().__init__(cfg_path, output_path, poll_interval, **kwargs)

        def _get_prof_args(self):
            """ Get prof_args py38"""
            return DynamicProfilerArgs.from_bytes(self._shm.buf[:DynamicProfilerArgs.SIZE])

        def _clean_resource(self):
            """Clean resource py38"""
            # stop profiler when stop_step over all train step
            if self._profiler:
                self._profiler.stop()
                self._profiler._ascend_profiler.finalize()
                self._profiler = None
                logger.warning("Rank %d Dynamic profiler stop at end of training", self._rank_id)

            # join process
            if self._process:
                self._shared_loop_flag.value = False
                self._process.join()
                self._process = None
                logger.info("Rank %s process stop", self._rank_id)

            # clear shared memory
            if self._shm:
                try:
                    self._shm.close()
                    self._shm.unlink()
                    logger.info("Rank %s unlink shm", self._rank_id)
                except FileNotFoundError:
                    logger.warning("Rank %s unlink shm failed, may be removed", self._rank_id)
                self._shm = None

        def _create_shm(self):
            """Create a json monitor process based on whether the SharedMemory is successfully created py38"""
            try_times = 10
            while try_times:
                try:
                    # Step 1: try to open shm file, first time shm not exists.
                    # Python incorrectly tracks shared memory even if it is not
                    # created by the process. The following patch is a workaround.
                    with patch("multiprocessing.resource_tracker.register",
                               lambda *args, **kwargs: None):
                        self._shm = shared_memory.SharedMemory(name=self._shm_name)
                    self._is_create_process = False
                    logger.info("Rank %d shared memory is connected.", self._rank_id)
                    break
                except FileNotFoundError:
                    try:
                        # Step 2: only one process can create shm successfully.
                        self._shm = shared_memory.SharedMemory(name=self._shm_name,
                                                               create=True, size=DynamicProfilerArgs.SIZE)
                        self._is_create_process = True
                        logger.info("Rank %d shared memory is created.", self._rank_id)
                        break
                    except FileExistsError:
                        # other process will go to step 1 and open shm file
                        try_times -= 1
                        logger.warning("Rank %d shared memory create failed, "
                                       "retry times = %d.", self._rank_id, try_times)
                        time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms

            if try_times <= 0:
                raise RuntimeError(f"Rank {self._rank_id} failed to create shared memory.")

else:
    import mmap
    import stat

    class DynamicProfilerMonitor(DynamicProfilerMonitorBase):
        r"""
        This class to enable the dynamic profile monitoring of MindSpore neural networks.

        Args:
            cfg_path (str): Dynamic profile json config file directory. The requirement is a shared path
                that can be accessed by all nodes.
            output_path (str, optional): Output data path. Default: ``"./dyn_profile_data"`` .
            poll_interval (int, optional): The polling period of the monitoring process, in seconds.
                Default value: ``2``.

        Raises:
            RuntimeError: When create shared memory times exceeds max times.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore.profiler import DynamicProfilerMonitor
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
            ...     dynprof_cb = DynamicProfilerMonitor(cfg_path="./dyn_cfg", output_path="./dyn_prof_data")
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # register DynamicProfilerMonitor to model.train()
            ...     model.train(10, data, callbacks=[dynprof_cb])
        """

        def __init__(self, cfg_path, output_path="./dyn_profile_data", poll_interval=2, **kwargs):
            if not isinstance(cfg_path, str):
                raise TypeError("The cfg_path must be a string.")
            if not isinstance(output_path, str):
                logger.warning(f"The output_path must be a string, "
                               f"but got type {type(output_path)}, it will be set to './dyn_profile_data'.")
                output_path = "./dyn_profile_data"
            self._cfg_path = cfg_path
            self._shm_name = time.strftime("DynamicProfileShm%Y%m%d%H", time.localtime())
            self._shm_dir = os.path.join(self._cfg_path, "shm")
            PathManager.make_dir_safety(self._shm_dir)
            self._shm_path = os.path.realpath(os.path.join(self._shm_dir, self._shm_name))

            super().__init__(cfg_path, output_path, poll_interval, **kwargs)
            logger.warning("Dynamic profiler is not work well on python 3.7x, "
                           "please update to python 3.8+ for better performance.")

        def _get_prof_args(self):
            """ Get prof_args py37"""
            self._shm.seek(0)
            return DynamicProfilerArgs.from_bytes(self._shm.read(DynamicProfilerArgs.SIZE))

        def _clean_resource(self):
            """Clean resource py37"""
            # stop profiler when stop_step over all train step
            if self._profiler:
                self._profiler.stop()
                self._profiler._ascend_profiler.finalize()
                self._profiler = None
                logger.warning("Rank %d Dynamic profiler stop at end of training", self._rank_id)

            # join process
            if self._process:
                self._shared_loop_flag.value = False
                self._process.join()
                self._process = None
                logger.info("Rank %s process stop", self._rank_id)

            # clear shared memory
            if self._shm and self._is_create_process:
                try:
                    self._shm.close()
                    if self._memory_mapped_file and not self._memory_mapped_file.closed:
                        self._memory_mapped_file.close()
                    elif self.fd:
                        os.close(self.fd)
                    PathManager.remove_file_safety(self._shm_path)
                    logger.info("Rank %s unlink shm", self._rank_id)
                except FileNotFoundError:
                    logger.warning("Rank %s unlink shm failed, may be removed", self._rank_id)
                self._shm = None

        def _create_shm(self):
            """Create a json monitor process based on whether the SharedMemory is successfully created py37"""

            try_times = 10
            while try_times:
                try:
                    # Step 1: try to open fd, first time fd not exists.
                    self.fd = os.open(self._shm_path, os.O_EXCL | os.O_RDWR,
                                      stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
                    self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                    self._shm = mmap.mmap(self._memory_mapped_file.fileno(), length=DynamicProfilerArgs.SIZE)
                    self._is_create_process = False
                    logger.info("Rank %d shared memory is connected.", self._rank_id)
                    break
                except ValueError:
                    time.sleep(0.02)
                except FileNotFoundError:
                    try:
                        # Step 2: only one process can create fd successfully.
                        fd = os.open(self._shm_path, os.O_CREAT | os.O_EXCL | os.O_RDWR,
                                     stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)

                        # Init mmap file need to write data
                        with os.fdopen(fd, 'wb') as f:
                            data_instance = DynamicProfilerArgs()
                            byte_data = data_instance.to_bytes()
                            f.write(byte_data)

                        # create mmap
                        self.fd = os.open(self._shm_path, os.O_EXCL | os.O_RDWR,
                                          stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP)
                        self._memory_mapped_file = os.fdopen(self.fd, 'rb')
                        self._shm = mmap.mmap(self._memory_mapped_file.fileno(), length=DynamicProfilerArgs.SIZE)
                        self._is_create_process = True
                        logger.info("Rank %d shared memory is created.", self._rank_id)
                        break
                    except FileExistsError:
                        # other process will go to step 1 and open shm file
                        try_times -= 1
                        logger.warning("Rank %d shared memory create failed, "
                                       "retry times = %d.", self._rank_id, try_times)
                        time.sleep(random.uniform(0, 0.02))  # sleep 0 ~ 20 ms

            if try_times <= 0:
                raise RuntimeError("Failed to create shared memory.")
