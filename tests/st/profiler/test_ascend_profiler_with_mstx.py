# Copyright 2025-2025 Huawei Technologies Co., Ltd
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
"""test ascend profiler with env mstx."""
import csv
import glob
import os
import tempfile
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import dataset as ds
from mindspore import context
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity
from model_zoo import TinyAddNet
from tests.mark_utils import arg_mark
from file_check import FileChecker

DATASET_PATH = "/home/workspace/mindspore_dataset/mnist"


def plot(imgs, first_origin=None):
    num_rows = 1
    num_cols = len(imgs)

    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for idx, img in enumerate(imgs):
        ax = axs[0, idx]
        ax.imshow(img.asnumpy())
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if first_origin:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    plt.tight_layout()


def check_result(result_path: str):
    expect_csv_header = ['Device_id', 'pid', 'tid', 'category', 'event_type', 'payload_type', 'payload_value',
                         'Start_time(us)', 'End_time(us)', 'message_type', 'message', 'domain', 'Device Start_time(us)',
                         'Device End_time(us)']
    expect_save_checkpoint_msg = 'save_checkpoint'
    expect_dataset_msg = 'dataloader'
    msprof_path = glob.glob(f'{result_path}/*_ascend_ms/PROF_*')[0]
    msproftx_csv_path = glob.glob(f'{msprof_path}/mindstudio_profiler_output/msprof_tx_*.csv')[0]
    current_result = []
    with open(msproftx_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            current_result.append(row)
    for i, value in enumerate(current_result):
        if i == 0:
            assert value == expect_csv_header
        elif i == 1:
            assert value[10] == expect_save_checkpoint_msg
        else:
            assert value[10] == expect_dataset_msg


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mstx_profiler():
    """
    Feature: Ascend Profiler for mstx data
    Description: Test Ascend Profiler for mstx data.
    Expectation: The profiler successfully collects data and generates the expected files.
    """
    with tempfile.TemporaryDirectory(suffix="_mstx_profiler") as tmpdir:
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

        mnist_ds = ds.MnistDataset(os.path.join(DATASET_PATH, "train"))

        net = TinyAddNet()
        prof = Profiler(activities=[ProfilerActivity.NPU],
                        mstx=True,
                        profiler_level=ProfilerLevel.LevelNone,
                        output_path=tmpdir,
                        data_simplification=False)

        ms.save_checkpoint(net, "./add.ckpt") # to get save checkpoint tx data

        # to get dataset tx data
        images = []
        for image, _ in mnist_ds:
            images.append(image)
            if len(images) > 5:
                break

        plot(images)

        prof.analyse()
        check_result(tmpdir)
        # Check profiler.log
        profiler_log_paths = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                       f"logs/profiler_*.log")
        for profiler_log_path in profiler_log_paths:
            FileChecker.check_file_for_keyword(profiler_log_path, "error")
