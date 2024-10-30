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
# ==============================================================================

import subprocess


def check_output(vlog_v, command, expect_output, is_expect=True):
    """set VLOG_v to vlog_v and check output """
    cmd = f"VLOG_v={vlog_v} python -c '{command}'"
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    out = s.stdout.read().decode("UTF-8")
    s.stdout.close()
    lines = out.split('\n')

    for output in expect_output:
        matched = False
        for line in lines:
            if line.find(output) > 0:
                matched = True
                break
        if is_expect:
            assert matched, f'`VLOG_v={vlog_v}` expect `{output}` fail'
        else:
            assert not matched, '`VLOG_v={vlog_v}` unexpected `{output}` fail'


def test_generator_perf():
    """
    Feature: Dataset perf statistics
    Description: Enable perf tp print data statistics of GeneratorDataset
    Expectation: Perf log can be printed normally
    """
    command = 'import mindspore.dataset as ds;'
    command += 'dataset = ds.GeneratorDataset([1, 2, 3, 4], ["A"]);'
    command += 'dataset = dataset.map(lambda x: x+1);'
    command += 'dataset = dataset.batch(2);'
    command += 'list(dataset)'

    expected = ['GeneratorOp', 'MapOp', 'BatchOp', 'worker_time', 'avg:', 'max:', 'min:']

    check_output('10900', command, expected, True)
    check_output('10901', command, expected, False)


def test_mindrecord_perf():
    """
    Feature: Dataset perf statistics
    Description: Enable perf tp print data statistics of MindDataset
    Expectation: Perf log can be printed normally
    """
    command = 'import mindspore.dataset as ds;'
    command += 'dataset = ds.MindDataset("../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0");'
    command += 'dataset = dataset.map(lambda x: x);'
    command += 'dataset = dataset.project(["data", "file_name"]);'
    command += 'dataset = dataset.batch(4);'
    command += 'list(dataset)'

    expected = ['MindRecordOp', 'MapOp', 'BatchOp', 'worker_time', 'io_time']
    unexpected = ['GeneratorOp']

    check_output('10900', command, expected, True)
    check_output('10900', command, unexpected, False)


if __name__ == '__main__':
    test_generator_perf()
    test_mindrecord_perf()
