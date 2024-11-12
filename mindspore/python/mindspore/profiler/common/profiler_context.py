# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""profiler context"""
import os
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Set, Callable,
)

from mindspore.communication.management import GlobalComm
from mindspore.communication.management import get_local_rank
from mindspore.communication.management import get_rank
from mindspore.profiler.common.constant import (
    DeviceTarget,
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics
)
from mindspore.profiler.common.profiler_output_path import ProfilerOutputPath
from mindspore.profiler.common.profiler_parameters import ProfilerParameters
from mindspore.profiler.common.singleton import Singleton
from mindspore.profiler.schedule import Schedule

from mindspore import context
from mindspore import log as logger


@Singleton
class ProfilerContext:
    """
    Profiler context manage all parameters and paths on runtime.
    """

    def __init__(self, **kwargs):
        self._profiler_params_mgr: ProfilerParameters = ProfilerParameters(**kwargs)
        self._device_id: Optional[str] = None
        self._rank_id: Optional[str] = None
        self._device_target: Optional[str] = None
        self._dynamic_status: Optional[bool] = None
        self._model_iteration_dict: Optional[Dict[int, int]] = None

        self._init_device_target()
        self._init_device_id()
        self._init_rank_id()
        self._profiler_path_mgr: ProfilerOutputPath = ProfilerOutputPath(
            device_id=int(self._device_id), rank_id=int(self._rank_id)
        )
        self._profiler_path_mgr.output_path = self._profiler_params_mgr.output_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profiler context to a dictionary for multiprocessing.
        """
        return {
            **self._profiler_params_mgr.original_params,
            **self._profiler_path_mgr.to_dict(),
            "device_id": self._device_id,
            "rank_id": self._rank_id,
            "device_target": self._device_target,
            "dynamic_status": self._dynamic_status,
            "model_iteration_dict": self._model_iteration_dict,
        }

    def load_offline_profiler_params(self, profiler_parameters: Dict[str, Any]) -> None:
        """
        Update profiler parameters from profiler_info.json
        """
        if not profiler_parameters:
            raise ValueError("Profiler parameters is empty")

        for param, (_, _) in self._profiler_params_mgr.PARAMS.items():
            if param in profiler_parameters:
                # 处理特殊类型的参数
                if param == "profiler_level":
                    value = ProfilerLevel(profiler_parameters[param])
                elif param == "aicore_metrics":
                    value = AicoreMetrics(profiler_parameters[param])
                elif param == "activities":
                    value = [ProfilerActivity(activity) for activity in profiler_parameters[param]]
                else:
                    value = profiler_parameters[param]

                setattr(self._profiler_params_mgr, param, value)
                print(f"load_offline_profiler_params: {param} = {value}")

        print(f"original_params : {self._profiler_params_mgr.original_params}")

    @property
    def device_target_set(self) -> Set[str]:
        """
        Get the device target set for ProfilerInterface initialization.

        CPU is always included in the list, device_target includes CPU、Ascend、GPU.
        """
        return set([DeviceTarget.CPU.value, self._device_target])

    @property
    def npu_profiler_params(self) -> Dict[str, Any]:
        """
        Get NPU profiler parameters for Ascend profiler cpp backend.

        Returns:
            Dict[str, Any]: A dictionary of NPU profiler parameters.
        """
        return self._profiler_params_mgr.npu_profiler_params

    @property
    def original_params(self) -> Dict[str, str]:
        """Get the original parameters from ProfilerParameters."""
        return self._profiler_params_mgr.original_params

    @property
    def output_path(self) -> str:
        """Get the output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.output_path

    @property
    def ascend_ms_dir(self) -> str:
        """Get the Ascend MS directory from ProfilerOutputPath."""
        return self._profiler_path_mgr.ascend_ms_dir

    @ascend_ms_dir.setter
    def ascend_ms_dir(self, value: str):
        """Set the Ascend MS directory to ProfilerOutputPath."""
        self._profiler_path_mgr.ascend_ms_dir = value

    @property
    def ascend_profiler_output_path(self) -> str:
        """Get the Ascend profiler output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.ascend_profiler_output_path

    @property
    def framework_path(self) -> str:
        """Get the framework path from ProfilerOutputPath."""
        return self._profiler_path_mgr.framework_path

    @property
    def msprof_profile_path(self) -> str:
        """Get the MSProf profile path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_path

    @msprof_profile_path.setter
    def msprof_profile_path(self, value: str):
        """Set the MSProf profile path to ProfilerOutputPath."""
        self._profiler_path_mgr.msprof_profile_path = value

    @property
    def msprof_profile_host_path(self) -> str:
        """Get the MSProf profile host path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_host_path

    @property
    def msprof_profile_device_path(self) -> str:
        """Get the MSProf profile device path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_device_path

    @property
    def msprof_profile_log_path(self) -> str:
        """Get the MSProf profile log path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_log_path

    @property
    def msprof_profile_output_path(self) -> str:
        """Get the MSProf profile output path from ProfilerOutputPath."""
        return self._profiler_path_mgr.msprof_profile_output_path

    @property
    def profiler_level(self) -> ProfilerLevel:
        """Get the profiler level from ProfilerParameters."""
        return self._profiler_params_mgr.profiler_level

    @property
    def activities(self) -> List[ProfilerActivity]:
        """Get the activities from ProfilerParameters."""
        return self._profiler_params_mgr.activities

    @property
    def profile_memory(self) -> bool:
        """Get the profile memory from ProfilerParameters."""
        return self._profiler_params_mgr.profile_memory

    @property
    def parallel_strategy(self) -> bool:
        """Get the parallel strategy from ProfilerParameters."""
        return self._profiler_params_mgr.parallel_strategy

    @property
    def start_profile(self) -> bool:
        """Get the start profile from ProfilerParameters."""
        return self._profiler_params_mgr.start_profile

    @property
    def aicore_metrics(self) -> int:
        """Get the aicore metrics from ProfilerParameters."""
        return self._profiler_params_mgr.aicore_metrics

    @property
    def l2_cache(self) -> bool:
        """Get the l2 cache from ProfilerParameters."""
        return self._profiler_params_mgr.l2_cache

    @property
    def hbm_ddr(self) -> bool:
        """Get the hbm ddr from ProfilerParameters."""
        return self._profiler_params_mgr.hbm_ddr

    @property
    def pcie(self) -> bool:
        """Get the pcie from ProfilerParameters."""
        return self._profiler_params_mgr.pcie

    @property
    def sync_enable(self) -> bool:
        """Get the sync enable from ProfilerParameters."""
        return self._profiler_params_mgr.sync_enable

    @property
    def data_process(self) -> bool:
        """Get the data process from ProfilerParameters."""
        return self._profiler_params_mgr.data_process

    @property
    def with_stack(self) -> bool:
        """Get the with stack from ProfilerParameters."""
        return self._profiler_params_mgr.with_stack

    @property
    def data_simplification(self) -> bool:
        """Get the data simplification from ProfilerParameters."""
        return self._profiler_params_mgr.data_simplification

    @property
    def device_target(self) -> str:
        """Get device target."""
        return self._device_target

    @property
    def rank_id(self) -> str:
        """Get rank id."""
        return self._rank_id

    @rank_id.setter
    def rank_id(self, value: str) -> None:
        """Set rank id."""
        if not value:
            raise ValueError("Rank id must be a non-empty string")

        if not value.isdigit():
            raise ValueError("Rank id must be a number")
        self._rank_id = value

    @property
    def device_id(self) -> str:
        """Get device id."""
        return self._device_id

    @device_id.setter
    def device_id(self, value) -> None:
        """Set device id."""
        if not value:
            raise ValueError("Device id must be a non-empty string")

        if not value.isdigit():
            raise ValueError("Device id must be a number")
        self._device_id = value

    @property
    def dynamic_status(self) -> bool:
        """Get dynamic status."""
        return self._dynamic_status

    @property
    def model_iteration_dict(self) -> Dict[int, int]:
        """Get model iteration dict."""
        return self._model_iteration_dict

    @model_iteration_dict.setter
    def model_iteration_dict(self, value: Dict[int, int]):
        """Set the model iteration dict."""
        if not isinstance(value, dict):
            raise ValueError("model_iteration_dict must be a dictionary")
        self._model_iteration_dict = value

    def _init_device_target(self) -> None:
        """
        Initialize the device target.

        Raises:
            RuntimeError: If the device target is not supported.
        """
        self._device_target = context.get_context("device_target")

        if self._device_target and self._device_target not in (
                member.value for member in DeviceTarget
        ):
            msg = "Profiling: unsupported backend: %s" % self._device_target
            raise RuntimeError(msg)

    def _init_device_id(self) -> None:
        """
        Initialize the device ID.
        """
        self._device_id = str(context.get_context("device_id"))

        if not self._device_id or not self._device_id.isdigit():
            if GlobalComm.INITED and self._device_target == DeviceTarget.NPU.value:
                self._device_id = str(get_local_rank())
            else:
                self._device_id = os.getenv("DEVICE_ID")

        if not self._device_id or not self._device_id.isdigit():
            self._device_id = "0"
            logger.warning("Fail to get DEVICE_ID, use 0 instead.")

    def _init_rank_id(self) -> None:
        """
        Initialize the rank ID.
        """
        if GlobalComm.INITED and self._device_target == DeviceTarget.NPU.value:
            self._rank_id = str(get_rank())
        else:
            self._rank_id = os.getenv("RANK_ID")

        if not self._rank_id or not self._rank_id.isdigit():
            self._rank_id = "0"
            logger.warning("Fail to get RANK_ID, use 0 instead.")

    @property
    def schedule(self) -> Schedule:
        return self._profiler_params_mgr.schedule

    @property
    def on_trace_ready(self) -> Optional[Callable[..., Any]]:
        return self._profiler_params_mgr.on_trace_ready
