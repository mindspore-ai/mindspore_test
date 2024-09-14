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
import os
import shutil
import tempfile

from mindspore import nn, context
from mindspore.nn.optim import Momentum
from mindspore.train import Model, Accuracy
from mindspore import Profiler
from tests.mark_utils import arg_mark
from model_zoo import LeNet5
from fake_dataset import FakeDataset
from file_check import FileChecker


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gpu_profiler():
    """
    Feature: profiler support GPU  mode.
    Description: profiling op time and timeline.
    Expectation: No exception.
    """
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    with tempfile.TemporaryDirectory(suffix="profiler_data") as data_path:
        profiler_path = os.path.join(data_path, 'profiler/')
        _train_with_profiler(data_path=data_path, device_target="GPU", profile_memory=False,
                             context_mode=context.GRAPH_MODE)
        _check_gpu_profiling_file(profiler_path, device_id)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gpu_profiler_pynative():
    """
    Feature: profiler support GPU pynative mode.
    Description: profiling l2 GPU pynative mode data, analyze performance issues.
    Expectation: No exception.
    """
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    with tempfile.TemporaryDirectory(suffix="profiler_data") as data_path:
        profiler_path = os.path.join(data_path, 'profiler/')
        _train_with_profiler(data_path=data_path, device_target="GPU", profile_memory=False,
                             context_mode=context.PYNATIVE_MODE)
        _check_gpu_profiling_file(profiler_path, device_id)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_gpu_profiler_env():
    """
    Feature: profiler support GPU  mode start by, enabling profiling through environment variables.
    Description: profiling op time and timeline.
    Expectation: No exception.
    """
    root_status = os.system("whoami | grep root")
    cuda_status = os.system("nvcc -V | grep 'release 10'")
    if root_status and not cuda_status:
        return
    status = os.system(
        """export MS_PROFILER_OPTIONS='{"start":true, "profile_framework":"all", "profile_memory":true, "sync_enable":true, "data_process":true}';
           python ./run_net.py --target=GPU --mode=0;
        """
    )
    data_path = os.path.join(os.getcwd(), 'data')
    try:
        profiler_path = os.path.join(os.getcwd(), 'data/profiler/')
        _check_gpu_profiling_file(profiler_path, 0)
        assert status == 0
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_gpu_profiler_pynative_env():
    """
    Feature: profiler support GPU pynative mode, enabling profiling through environment variables.
    Description: profiling l2 GPU pynative mode data, analyze performance issues.
    Expectation: No exception.
    """
    root_status = os.system("whoami | grep root")
    cuda_status = os.system("nvcc -V | grep 'release 10'")
    if root_status and not cuda_status:
        return
    status = os.system(
        """export MS_PROFILER_OPTIONS='{"start":true, "profile_framework":"all", "sync_enable":true, "data_process":true}';
           python ./run_net.py --target=GPU --mode=1;
        """
    )
    data_path = os.path.join(os.getcwd(), 'data')
    try:
        profiler_path = os.path.join(os.getcwd(), 'data/profiler/')
        _check_gpu_profiling_file(profiler_path, 0)
        assert status == 0
    finally:
        if os.path.exists(data_path):
            shutil.rmtree(data_path)


def _train_with_profiler(device_target, profile_memory, data_path, context_mode=context.GRAPH_MODE,
                         profile_framework='all'):
    context.set_context(mode=context_mode, device_target=device_target)
    ds_train = FakeDataset.create_fake_cv_dataset()
    if ds_train.get_dataset_size() == 0:
        raise ValueError("Please check dataset size > 0 and batch_size <= dataset size")

    profiler = Profiler(profile_memory=profile_memory, output_path=data_path,
                        profile_framework=profile_framework, data_simplification=False,
                        data_process=True)
    lenet = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})

    model.train(1, ds_train, dataset_sink_mode=True)
    profiler.analyse()
    profiler.op_analyse(op_name="Conv2D")


def _check_gpu_profiling_file(profiler_path, device_id):
    op_detail_file = profiler_path + f'gpu_op_detail_info_{device_id}.csv'
    op_type_file = profiler_path + f'gpu_op_type_info_{device_id}.csv'
    activity_file = profiler_path + f'gpu_activity_data_{device_id}.csv'
    timeline_file = profiler_path + f'gpu_timeline_display_{device_id}.json'
    getnext_file = profiler_path + f'minddata_getnext_profiling_{device_id}.txt'
    pipeline_file = profiler_path + f'minddata_pipeline_raw_{device_id}.csv'
    framework_file = profiler_path + f'gpu_framework_{device_id}.txt'
    dataset_csv = os.path.join(profiler_path, f'dataset_{device_id}.csv')

    op_dict = {"op_full_name": ["*Conv2D*", "*MatMul*"], "op_type": ["BiasAddGrad"]}
    FileChecker.check_csv_items(op_detail_file, op_dict, fuzzy_match=True)
    op_dict = {"op_type": "MaxPoolGrad"}
    FileChecker.check_csv_items(op_type_file, op_dict, fuzzy_match=True)
    op_dict = {"op_full_name": ["*MatMul*", "*Conv2D*"]}
    FileChecker.check_csv_items(activity_file, op_dict, fuzzy_match=True)
    FileChecker.check_timeline_values(timeline_file, "name", ["cudnn::maxpooling*", "MaxPool*"], True)
    FileChecker.check_file_line_count(getnext_file, 10)
    FileChecker.check_file_line_count(pipeline_file, 9)
    FileChecker.check_txt_not_empty(framework_file)
    op_dict = {"Operation": ["DataQueueOp*", "MapOp*"]}
    FileChecker.check_csv_items(dataset_csv, op_dict, fuzzy_match=True)
