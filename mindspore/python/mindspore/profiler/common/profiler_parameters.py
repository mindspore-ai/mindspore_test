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
# ===========================================================================
"""ProfilerParameters"""
import warnings
from typing import Dict, Optional, Callable, Any

from mindspore import log as logger
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics,
)
from mindspore.profiler.schedule import Schedule


class ProfilerParameters:
    """
    Profiler parameters manage all parameters, parameters validation and type conversion.
    """

    # key: Parameter name, value: (type, default value)
    PARAMS: Dict[str, tuple] = {
        "output_path": (str, "./data"),
        "profiler_level": (ProfilerLevel, ProfilerLevel.Level0),
        "activities": (list, [ProfilerActivity.CPU, ProfilerActivity.NPU]),
        "aicore_metrics": (AicoreMetrics, AicoreMetrics.AiCoreNone),
        "with_stack": (bool, False),
        "profile_memory": (bool, False),
        "data_process": (bool, False),
        "parallel_strategy": (bool, False),
        "start_profile": (bool, True),
        "l2_cache": (bool, False),
        "hbm_ddr": (bool, False),
        "pcie": (bool, False),
        "sync_enable": (bool, True),
        "data_simplification": (bool, True),
        "mstx": (bool, False),
        "schedule": (Schedule, None),
        "on_trace_ready": (Optional[Callable[..., Any]], None)
    }

    TYPE_INDEX = 0
    VALUE_INDEX = 1

    def __init__(self, **kwargs):
        self.is_set_schedule: bool = False
        self._set_schedule(**kwargs)
        self._check_deprecated_params(**kwargs)
        # Initialize parameters with kwargs
        for param, (_, default_value) in self.PARAMS.items():
            setattr(self, param, kwargs.get(param, default_value))

        self._check_params_type()
        self._handle_compatibility()

    @property
    def original_params(self) -> Dict[str, str]:
        """
        Get params dict for profiler_info.json save.
        """
        params = {}
        params["is_set_schedule"] = self.is_set_schedule
        for param, (_, _) in self.PARAMS.items():
            if param == "profiler_level":
                params[param] = getattr(self, param).value
            elif param == "aicore_metrics":
                params[param] = getattr(self, param).value
            elif param == "activities":
                params[param] = [item.value for item in getattr(self, param)]
            elif param == "schedule":
                params[param] = getattr(self, param).to_dict()
            elif param == "on_trace_ready":
                continue
            else:
                params[param] = getattr(self, param)
        return params

    @property
    def npu_profiler_params(self) -> Dict[str, str]:
        """
        Get NPU profiler parameters for Ascend profiler cpp backend.

        Returns:
            Dict[str, str]: A dictionary of NPU profiler parameters.
        """
        return {
            "profile_memory": self.profile_memory,
            "aic_metrics": self.aicore_metrics.value,
            "l2_cache": self.l2_cache,
            "hbm_ddr": self.hbm_ddr,
            "pcie": self.pcie,
            "parallel_strategy": self.parallel_strategy,
            "profiler_level": self.profiler_level.value,
            "aicore_metrics": self.aicore_metrics.value,
            "with_stack": self.with_stack,
            "mstx": self.mstx,
            "cpu_trace": ProfilerActivity.CPU in self.activities,
            "npu_trace": ProfilerActivity.NPU in self.activities,
        }

    def _check_params_type(self) -> None:
        """
        Check profiler input params type, if type is invalid reset to default value.
        """
        for key, value in self.__dict__.items():
            if key in ProfilerParameters.PARAMS:
                expected_type = ProfilerParameters.PARAMS[key][ProfilerParameters.TYPE_INDEX]
                default_value = ProfilerParameters.PARAMS[key][ProfilerParameters.VALUE_INDEX]

                # Callable特殊处理
                if key == "on_trace_ready":
                    if not callable(value):
                        setattr(self, key, default_value)
                elif key == "schedule":
                    if not isinstance(value, Schedule):
                        setattr(self, key, Schedule(wait=0, active=1))
                # 检查可迭代类型
                elif isinstance(expected_type, type) and issubclass(expected_type, (list, tuple, set)):
                    if not (isinstance(value, expected_type) and
                            all(isinstance(item, type(default_value[0])) for item in value)):
                        logger.warning(
                            f"For Profiler, {key} value is Invalid, reset to {default_value}."
                        )
                        setattr(self, key, default_value)
                # 检查普通类型
                elif not isinstance(value, expected_type):
                    logger.warning(
                        f"For Profiler, the type of {key} should be {expected_type}, "
                        f"but got {type(value)}, reset to {default_value}."
                    )
                    setattr(self, key, default_value)

    def _check_deprecated_params(self, **kwargs) -> None:
        """
        Check deprecated parameters.
        """
        for key, _ in kwargs.items():
            if key == "profile_communication":
                warnings.warn(
                    "The parameter 'profile_communication' is deprecated,"
                    " please use 'profiler_level=ProfilerLevel.Level1' or "
                    "'profiler_level=ProfilerLevel.Level2' instead."
                )
            elif key == "op_time":
                warnings.warn(
                    "The parameter 'op_time' is deprecated,"
                    " please use 'activaties=ProfilerActivity.NPU' instead."
                )
            elif key == "profile_framework":
                warnings.warn(
                    "The parameter 'profile_framework' is deprecated,"
                    " please use 'activaties=ProfilerActivity.CPU' instead."
                )
            elif key == "host_stack":
                warnings.warn(
                    "The parameter 'host_stack' is deprecated,"
                    " please use 'with_stack' instead."
                )
            elif key == "timeline_limit":
                warnings.warn(
                    "The parameter 'timeline_limit' is deprecated and will have no effect"
                )

    def _set_schedule(self, **kwargs):
        if "schedule" in kwargs and isinstance(kwargs["schedule"], Schedule):
            self.is_set_schedule = True

    def _handle_compatibility(self) -> None:
        """
        Handle compatibility.
        """
        if hasattr(self, "schedule") and self.is_set_schedule and self.__dict__.get('data_process', False):
            self.data_process = False
            warnings.warn("When 'schedule' is set, 'data_process' will be set to False.")

        if not self.__dict__.get('mstx') and self.__dict__.get('profiler_level') == ProfilerLevel.LevelNone:
            self.profiler_level = ProfilerLevel.Level0
            warnings.warn("when 'mstx' is disabled, 'profiler_level' cannot be set to 'ProfilerLevel.LevelNone', "
                          "reset to 'ProfilerLevel.Level0'.")

        if self.__dict__.get('profiler_level') == ProfilerLevel.Level0 and \
            self.__dict__.get('aicore_metrics') != AicoreMetrics.AiCoreNone:
            self.aicore_metrics = AicoreMetrics.AiCoreNone
            warnings.warn("when 'profiler_level' is set to 'ProfilerLevel.Level0', "
                          "'aicore_metrics' cannot be set to other value except 'AicoreMetrics.AiCoreNone', "
                          "reset to 'AicoreMetrics.AiCoreNone'.")

        if self.__dict__.get('profiler_level') != ProfilerLevel.Level0 and \
            self.__dict__.get('aicore_metrics') == AicoreMetrics.AiCoreNone:
            self.aicore_metrics = AicoreMetrics.PipeUtilization

    def __getattr__(self, name):
        """
        Get attribute.
        """
        if name in self.PARAMS:
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
