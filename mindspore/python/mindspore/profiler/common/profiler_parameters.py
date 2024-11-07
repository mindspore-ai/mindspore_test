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
from typing import Dict
import warnings

from mindspore import log as logger
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics,
)


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
    }

    TYPE_INDEX = 0
    VALUE_INDEX = 1
    ENABLE_STATUS = "on"
    DISABLE_STATUS = "off"

    def __init__(self, **kwargs):
        # Initialize parameters with kwargs
        for param, (_, default_value) in self.PARAMS.items():
            setattr(self, param, kwargs.get(param, default_value))

        self._check_params_type()
        self._check_deprecated_params()

    @property
    def original_params(self) -> Dict[str, str]:
        """
        Get params dict for profiler_info.json save.
        """
        params = {}
        for param, (_, _) in self.PARAMS.items():
            if param == "profiler_level":
                params[param] = getattr(self, param).value
            elif param == "aicore_metrics":
                params[param] = getattr(self, param).value
            elif param == "activities":
                params[param] = [item.value for item in getattr(self, param)]
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
            "output": self.output_path,
            "training_trace": self._convert_bool_to_status(ProfilerActivity.NPU in self.activities),
            "aic_metrics": self.aicore_metrics.value,
            "profile_memory": self._convert_bool_to_status(
                self.profile_memory and ProfilerActivity.NPU in self.activities
            ),
            "l2_cache": self._convert_bool_to_status(
                self.l2_cache and ProfilerActivity.NPU in self.activities
            ),
            "hbm_ddr": self._convert_bool_to_status(
                self.hbm_ddr and ProfilerActivity.NPU in self.activities
            ),
            "pcie": self._convert_bool_to_status(
                self.pcie and ProfilerActivity.NPU in self.activities
            ),
            "parallel_strategy": self._convert_bool_to_status(self.parallel_strategy),
            "profiler_level": (
                self.profiler_level.value
                if self.profiler_level and ProfilerActivity.NPU in self.activities
                else self.DISABLE_STATUS
            ),
            "with_stack": self._convert_bool_to_status(
                self.with_stack and ProfilerActivity.CPU in self.activities
            ),
        }

    def _check_params_type(self) -> None:
        """
        Check profiler input params type, if type is invalid reset to default value.
        """
        for key, value in self.__dict__.items():
            if key in ProfilerParameters.PARAMS:
                expected_type = ProfilerParameters.PARAMS[key][ProfilerParameters.TYPE_INDEX]
                default_value = ProfilerParameters.PARAMS[key][ProfilerParameters.VALUE_INDEX]

                # 检查可迭代类型
                if isinstance(expected_type, type) and issubclass(expected_type, (list, tuple, set)):
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

    def _check_deprecated_params(self) -> None:
        """
        Check deprecated parameters.
        """
        for key, _ in self.__dict__.items():
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

    def __getattr__(self, name):
        """
        Get attribute.
        """
        if name in self.PARAMS:
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def _convert_bool_to_status(self, condition):
        """
        Convert bool to on or off string for npu profiler cpp backend.
        """
        return self.ENABLE_STATUS if condition else self.DISABLE_STATUS
