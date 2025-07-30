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

import tempfile
import os
import json

from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap
from cmp_dump_statistic import compare_csv_files
from pathlib import Path


def generate_dump_json(dump_path, json_file_name, enable_sync):
    current_dir = Path(__file__).parent.parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
        data["common_dump_settings"]["path"] = dump_path
        data["e2e_dump_settings"]["enable"] = enable_sync
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@security_off_wrap
def test_multi_stream_async_statistic_dump():
    """
    Feature: Multi-stream async dump statistic
    Description: Test the multi-stream async dump statistic functionality.
    Expectation: The test should pass without any errors.
    Steps:
        1. Generate JSON configuration files for sync and async dumps.
        2. Execute the dump script for sync and async dumps.
        3. Compare the results of sync and async dumps.
    """
    sh_path = str(Path(__file__).parent.absolute())
    data_path = "/home/workspace/mindspore_dataset/mnist/train/"

    with tempfile.TemporaryDirectory() as test_dir:
        path = Path(test_dir).absolute()
        def exec_dump_and_get_dump_path(base_path, name, enable_sync):
            """Execute dump and return the path."""
            dump_path = str(base_path / f"{name}_data")
            dump_config_path = str(base_path / f"{name}_config.json")
            generate_dump_json(dump_path, dump_config_path, enable_sync)
            ret = os.system(f"bash {sh_path}/msrun_single.sh {data_path} {dump_config_path}")
            assert ret == 0, f"{name} exec failed"
            return dump_path

        sync_dump_path = exec_dump_and_get_dump_path(path, "sync_dump", True)
        async_dump_path = exec_dump_and_get_dump_path(path, "async_dump", False)

        # compare result
        compare_csv_files(sync_dump_path, async_dump_path)
