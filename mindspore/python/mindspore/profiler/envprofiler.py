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
"""EnvProfiler"""
import os
import json
from mindspore import log as logger
from mindspore.profiler.profiler import Profiler
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    AicoreMetrics,
    ProfilerActivity,
)
from mindspore.profiler.common.profiler_parameters import ProfilerParameters


class EnvProfiler:
    """Collect and analyze training performance data, support calls during and after training."""

    NOT_SUPPORTED_PARAMS = ["schedule", "on_trace_ready"]
    profiler = None

    @classmethod
    def init_profiler(cls):
        """
        Initialize the profiler.
        """
        if not os.getenv("MS_PROFILER_OPTIONS"):
            return
        options = cls._load_options()
        if not options:
            logger.error("Failed to load MS_PROFILER_OPTIONS, json decode error.")
            return

        params = cls._convert_options_to_profiler_params(options)
        logger.info(f"params: {params}")
        if params["start_profile"]:
            cls.profiler = Profiler(**params)
            logger.info("Profiler init success.")

    def analyse(self):
        """
        Analyze the collected data.
        """
        logger.info("analyse start")
        if not self.profiler:
            logger.info("Profiler is not initialized, skip analyse.")
            return
        self.profiler.analyse()
        logger.info("analyse end")

    @classmethod
    def _load_options(cls):
        """
        Load the options from the environment variable.
        """
        try:
            options = json.loads(os.environ.get("MS_PROFILER_OPTIONS", "{}"))
        except json.JSONDecodeError:
            return {}
        return options

    @classmethod
    def _convert_option_to_enum_value(cls, enum_class, option_value, default_value):
        """
        Convert the option value to the enum value.
        """
        try:
            return enum_class(option_value)
        except ValueError:
            logger.warning(
                f"The value '{option_value}' of parameter '{enum_class.__name__}' is invalid, "
                f"use default value '{default_value}' instead."
            )
            return default_value

    @classmethod
    def _convert_options_to_profiler_params(cls, options):
        """
        Convert the options to the profiler parameters.
        """
        params = {}
        if not options:
            logger.warning("MS_PROFILER_OPTIONS is empty, use default values.")
            return params

        if "output_path" in options:
            params["output_path"] = options["output_path"]

        # if start is not set, default is False
        params["start_profile"] = options.get("start", False)

        for param, (_, default_value) in ProfilerParameters.PARAMS.items():
            if param in options and param not in cls.NOT_SUPPORTED_PARAMS:
                if param == "activities" and isinstance(options[param], list):
                    params[param] = cls._convert_activities_to_list(
                        options[param], default_value
                    )
                elif param == "aicore_metrics":
                    params[param] = cls._convert_option_to_enum_value(
                        AicoreMetrics, options[param], default_value
                    )
                elif param == "profiler_level":
                    params[param] = cls._convert_option_to_enum_value(
                        ProfilerLevel, options[param], default_value
                    )
                else:
                    params[param] = options[param]
        return params

    @classmethod
    def _convert_activities_to_list(cls, activities, default_value):
        """
        Convert the activities to the list.
        """
        res = []
        for activity in activities:
            res.append(
                cls._convert_option_to_enum_value(
                    ProfilerActivity, activity, default_value
                )
            )
        # remove duplicate
        return list(set(default_value if default_value in res else res))


EnvProfiler.init_profiler()
