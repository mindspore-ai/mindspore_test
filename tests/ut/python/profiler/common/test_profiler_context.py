# pylint: disable=protected-access
import os
import unittest
from mindspore.profiler.common.constant import (
    ProfilerLevel,
    ProfilerActivity,
    AicoreMetrics,
    DeviceTarget,
)
from mindspore.profiler.common.profiler_context import ProfilerContext


class TestProfilerContext(unittest.TestCase):
    """Test the profiler context."""

    def setUp(self):
        ProfilerContext._instance = {}

    def tearDown(self):
        ProfilerContext._instance = {}

    def test_profiler_context_should_set_correct_values_when_valid_kwargs(self):
        """Test the valid kwargs."""
        valid_kwargs = {
            "output_path": "./profiler_data",
            "profiler_level": ProfilerLevel.Level0,
            "activities": [ProfilerActivity.CPU],
            "profile_memory": False,
            "parallel_strategy": False,
            "start_profile": True,
            "aicore_metrics": AicoreMetrics.PipeUtilization,
            "l2_cache": False,
            "hbm_ddr": False,
            "pcie": False,
            "sync_enable": True,
            "data_process": False,
            "with_stack": False,
            "data_simplification": True,
        }
        prof_ctx = ProfilerContext(**valid_kwargs)

        self.assertEqual(prof_ctx.output_path, valid_kwargs["output_path"])
        self.assertEqual(prof_ctx.profiler_level, valid_kwargs["profiler_level"])
        self.assertEqual(prof_ctx.activities, valid_kwargs["activities"])
        self.assertEqual(prof_ctx.profile_memory, valid_kwargs["profile_memory"])
        self.assertEqual(prof_ctx.parallel_strategy, valid_kwargs["parallel_strategy"])
        self.assertEqual(prof_ctx.start_profile, valid_kwargs["start_profile"])
        self.assertEqual(prof_ctx.aicore_metrics, valid_kwargs["aicore_metrics"])
        self.assertEqual(prof_ctx.l2_cache, valid_kwargs["l2_cache"])
        self.assertEqual(prof_ctx.hbm_ddr, valid_kwargs["hbm_ddr"])
        self.assertEqual(prof_ctx.pcie, valid_kwargs["pcie"])
        self.assertEqual(prof_ctx.sync_enable, valid_kwargs["sync_enable"])
        self.assertEqual(prof_ctx.data_process, valid_kwargs["data_process"])
        self.assertEqual(prof_ctx.with_stack, valid_kwargs["with_stack"])
        self.assertEqual(prof_ctx.data_simplification, valid_kwargs["data_simplification"])

    def test_profiler_context_should_set_default_values_when_invalid_kwargs(self):
        """Test the invalid kwargs."""
        invalid_kwargs = {
            "output_path": 1,
            "profiler_level": "O0",
            "activities": "x",
            "profile_memory": 1,
            "parallel_strategy": 1,
            "start_profile": 1,
            "aicore_metrics": "x",
            "l2_cache": 1,
            "hbm_ddr": 1,
            "pcie": 1,
            "sync_enable": 1,
            "data_process": 1,
            "with_stack": 1,
            "data_simplification": 1,
        }

        prof_ctx = ProfilerContext(**invalid_kwargs)
        self.assertEqual(prof_ctx.output_path, "./data")
        self.assertEqual(prof_ctx.profiler_level, ProfilerLevel.Level0)
        self.assertEqual(prof_ctx.activities, [ProfilerActivity.CPU, ProfilerActivity.NPU])
        self.assertEqual(prof_ctx.profile_memory, False)
        self.assertEqual(prof_ctx.parallel_strategy, False)
        self.assertEqual(prof_ctx.start_profile, True)
        self.assertEqual(prof_ctx.aicore_metrics, AicoreMetrics.AiCoreNone)
        self.assertEqual(prof_ctx.l2_cache, False)
        self.assertEqual(prof_ctx.hbm_ddr, False)
        self.assertEqual(prof_ctx.pcie, False)
        self.assertEqual(prof_ctx.sync_enable, True)
        self.assertEqual(prof_ctx.data_process, False)
        self.assertEqual(prof_ctx.with_stack, False)
        self.assertEqual(prof_ctx.data_simplification, True)

    def test_profiler_context_should_set_correct_values_when_profiler_output_path_properties(self):
        """Test the profiler output path properties."""
        output_path = "./data"
        ascend_ms_dir = "xxx_ascend_ms"
        msprof_profile_path = "PROF_XXX"
        prof_ctx = ProfilerContext()
        self.assertEqual(prof_ctx.output_path, output_path)

        prof_ctx.ascend_ms_dir = ascend_ms_dir
        self.assertEqual(prof_ctx.ascend_ms_dir, os.path.join(output_path, ascend_ms_dir))
        self.assertEqual(prof_ctx.framework_path, os.path.join(output_path, ascend_ms_dir, "FRAMEWORK"))
        prof_ctx.msprof_profile_path = os.path.join(output_path, ascend_ms_dir, msprof_profile_path)
        self.assertEqual(prof_ctx.msprof_profile_path, os.path.join(output_path, ascend_ms_dir, msprof_profile_path))
        self.assertEqual(
            prof_ctx.msprof_profile_host_path,
            os.path.join(output_path, ascend_ms_dir, msprof_profile_path, "host")
        )
        self.assertEqual(
            prof_ctx.msprof_profile_device_path,
            os.path.join(output_path, ascend_ms_dir, msprof_profile_path, "device_0")
        )
        self.assertEqual(
            prof_ctx.msprof_profile_log_path,
            os.path.join(output_path, ascend_ms_dir, msprof_profile_path, "mindstudio_profiler_log"),
        )
        self.assertEqual(
            prof_ctx.msprof_profile_output_path,
            os.path.join(output_path, ascend_ms_dir, msprof_profile_path, "mindstudio_profiler_output"),
        )

    def test_profiler_context_should_raise_value_error_when_profiler_output_path_properties_getattr_before_set(self):
        """Test the profiler output path properties get attribute error before set."""
        test_cases = [
            "msprof_profile_path",
            "msprof_profile_host_path",
            "msprof_profile_device_path",
            "msprof_profile_log_path",
            "msprof_profile_output_path",
        ]
        prof_ctx = ProfilerContext()
        with self.assertRaises(ValueError):
            for test_case in test_cases:
                getattr(prof_ctx, test_case)

    def test_npu_profiler_params_and_original_params_should_set_correct_values_when_valid_kwargs(self):
        """Test the npu_profiler_params and original_params properties."""
        prof_ctx = ProfilerContext(
            output_path="./profiler_data",
            profiler_level=ProfilerLevel.Level1,
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            profile_memory=True,
            parallel_strategy=True,
            start_profile=True,
            aicore_metrics=AicoreMetrics.PipeUtilization,
            l2_cache=True,
            hbm_ddr=True,
            pcie=True,
            sync_enable=True,
            data_process=True,
            with_stack=True,
            data_simplification=True,
        )
        self.assertEqual(
            prof_ctx.npu_profiler_params,
            {
                "output": "./profiler_data",
                "training_trace": "on",
                "aic_metrics": "PipeUtilization",
                "profile_memory": "on",
                "l2_cache": "on",
                "hbm_ddr": "on",
                "pcie": "on",
                "parallel_strategy": "on",
                "profiler_level": "Level1",
                "with_stack": "on",
            },
        )
        self.assertEqual(
            prof_ctx.original_params,
            {
                "output_path": "./profiler_data",
                "profiler_level": "Level1",
                "activities": ["CPU", "NPU"],
                "aicore_metrics": "PipeUtilization",
                "with_stack": True,
                "profile_memory": True,
                "data_process": True,
                "parallel_strategy": True,
                "start_profile": True,
                "l2_cache": True,
                "hbm_ddr": True,
                "pcie": True,
                "sync_enable": True,
                "data_simplification": True,
            },
        )

    def test_reset_method_should_set_correct_values_when_output_path_is_valid(self):
        """Test the reset method of the singleton class."""
        prof_cxt1 = ProfilerContext(output_path="./profiler_data")
        self.assertEqual(prof_cxt1.output_path, "./profiler_data")
        prof_cxt2 = ProfilerContext.reset(output_path="./profiler_data2")
        self.assertEqual(prof_cxt2.output_path, "./profiler_data2")

    def test_device_target_set_property_should_set_correct_values_when_device_target_is_valid(self):
        """Test the device_target_set property."""
        prof_ctx = ProfilerContext()
        prof_ctx._device_target = DeviceTarget.NPU.value
        self.assertEqual(prof_ctx.device_target_set, set([DeviceTarget.CPU.value, DeviceTarget.NPU.value]))
        prof_ctx._device_target = DeviceTarget.CPU.value
        self.assertEqual(prof_ctx.device_target_set, set([DeviceTarget.CPU.value]))
