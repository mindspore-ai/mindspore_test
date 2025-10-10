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
"""Test adaptive checkpoint functionality."""
import os
import tempfile

import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal
from mindspore.train.callback._adaptive_checkpoint import AdaptiveCheckpointConfig, AdaptiveModelCheckpoint
from mindspore.train import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim import SGD
from tests.mark_utils import arg_mark


class SimpleNet(nn.Cell):
    """Simple network for testing."""

    def __init__(self, num_class=10, num_channel=1):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        """Forward network."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataset(batch_size=32, num_samples=128):
    """Create a simple dataset for testing."""
    data = np.random.randn(num_samples, 1, 32, 32).astype(np.float32)
    labels = np.random.randint(0, 10, (num_samples,)).astype(np.int32)

    dataset = ms.dataset.NumpySlicesDataset((data, labels), column_names=["data", "label"])
    dataset = dataset.batch(batch_size)
    return dataset


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_adaptive_checkpoint_config_basic(mode):
    """
    Feature: AdaptiveCheckpointConfig basic functionality.
    Description: Test basic configuration creation and validation.
    Expectation: Config should be created successfully with valid parameters.
    """
    context.set_context(mode=mode)

    config1 = AdaptiveCheckpointConfig(target_overhead_percentage=5.0)
    assert config1.target_overhead_percentage == 5.0
    assert config1.failure_rate is None
    assert config1.is_adaptive is True
    assert config1.adaptive_mode == "percentage"

    config2 = AdaptiveCheckpointConfig(failure_rate=0.001)
    assert config2.failure_rate == 0.001
    assert config2.target_overhead_percentage is None
    assert config2.is_adaptive is True
    assert config2.adaptive_mode == "failure_rate"

    config3 = AdaptiveCheckpointConfig()
    assert config3.target_overhead_percentage is None
    assert config3.failure_rate is None
    assert config3.is_adaptive is False
    assert config3.adaptive_mode is None


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_checkpoint_config_validation():
    """
    Feature: AdaptiveCheckpointConfig parameter validation.
    Description: Test parameter validation for invalid inputs.
    Expectation: Should raise appropriate ValueError for invalid parameters.
    """
    with pytest.raises(ValueError, match="only one of 'target_overhead_percentage' or 'failure_rate' can be set"):
        AdaptiveCheckpointConfig(target_overhead_percentage=5.0, failure_rate=0.001)

    with pytest.raises(ValueError, match="'target_overhead_percentage' must be a number in range"):
        AdaptiveCheckpointConfig(target_overhead_percentage=0)

    with pytest.raises(ValueError, match="'target_overhead_percentage' must be a number in range"):
        AdaptiveCheckpointConfig(target_overhead_percentage=101)

    with pytest.raises(ValueError, match="'target_overhead_percentage' must be a number in range"):
        AdaptiveCheckpointConfig(target_overhead_percentage=-5)

    with pytest.raises(ValueError, match="'failure_rate' must be a number in range"):
        AdaptiveCheckpointConfig(failure_rate=0)

    with pytest.raises(ValueError, match="'failure_rate' must be a number in range"):
        AdaptiveCheckpointConfig(failure_rate=1.5)

    with pytest.raises(ValueError, match="'failure_rate' must be a number in range"):
        AdaptiveCheckpointConfig(failure_rate=-0.1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_adaptive_model_checkpoint_basic(mode):
    """
    Feature: AdaptiveModelCheckpoint basic functionality.
    Description: Test basic checkpoint creation and configuration.
    Expectation: Checkpoint should be created successfully with valid config.
    """
    context.set_context(mode=mode)

    with tempfile.TemporaryDirectory() as temp_dir:
        config = AdaptiveCheckpointConfig(
            target_overhead_percentage=5.0,
            save_checkpoint_steps=10
        )
        checkpoint = AdaptiveModelCheckpoint(
            prefix='adaptive_test',
            directory=temp_dir,
            config=config
        )

        assert checkpoint._adaptive_config.target_overhead_percentage == 5.0  # pylint:disable=protected-access
        assert checkpoint._adaptive_config.save_checkpoint_steps == 10  # pylint:disable=protected-access
        assert checkpoint._adaptive_phase == "percentage"  # pylint:disable=protected-access
        assert checkpoint._current_save_interval == 10  # pylint:disable=protected-access


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adaptive_model_checkpoint_validation():
    """
    Feature: AdaptiveModelCheckpoint parameter validation.
    Description: Test validation for invalid configurations.
    Expectation: Should raise appropriate errors for invalid inputs.
    """
    with pytest.raises(TypeError, match="the type of argument 'config' should be 'AdaptiveCheckpointConfig'"):
        AdaptiveModelCheckpoint(config="invalid_config")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_adaptive_checkpoint_percentage_mode(mode):
    """
    Feature: AdaptiveModelCheckpoint percentage-based adaptation.
    Description: Test checkpoint saving with percentage-based adaptive interval.
    Expectation: Checkpoint files should be saved and interval should adapt.
    """
    context.set_context(mode=mode)

    with tempfile.TemporaryDirectory() as temp_dir:
        net = SimpleNet()
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = SGD(net.trainable_params(), learning_rate=0.01)

        config = AdaptiveCheckpointConfig(
            target_overhead_percentage=20.0,
            save_checkpoint_steps=5,
            keep_checkpoint_max=5,
            format="safetensors"
        )

        checkpoint = AdaptiveModelCheckpoint(
            prefix='percentage_test',
            directory=temp_dir,
            config=config
        )

        model = Model(net, loss_fn, optimizer)
        dataset = create_dataset(batch_size=8, num_samples=1024)

        model.train(1, dataset, callbacks=[checkpoint])

        ckpt_files = [f for f in os.listdir(temp_dir) if f.endswith('.safetensors')]
        assert ckpt_files

        param_dict = ms.load_checkpoint(os.path.join(temp_dir, ckpt_files[0]), format="safetensors")
        assert param_dict


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_adaptive_checkpoint_failure_rate_mode(mode):
    """
    Feature: AdaptiveModelCheckpoint failure rate-based adaptation.
    Description: Test checkpoint saving with failure rate-based adaptive interval.
    Expectation: Checkpoint files should be saved and interval should adapt.
    """
    context.set_context(mode=mode)

    with tempfile.TemporaryDirectory() as temp_dir:
        net = SimpleNet()
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = SGD(net.trainable_params(), learning_rate=0.01)

        config = AdaptiveCheckpointConfig(
            failure_rate=0.01,
            save_checkpoint_steps=5,
            keep_checkpoint_max=5
        )

        checkpoint = AdaptiveModelCheckpoint(
            prefix='failure_rate_test',
            directory=temp_dir,
            config=config
        )

        model = Model(net, loss_fn, optimizer)
        dataset = create_dataset(batch_size=8, num_samples=1024)

        model.train(1, dataset, callbacks=[checkpoint])

        ckpt_files = [f for f in os.listdir(temp_dir) if f.endswith('.ckpt')]
        assert ckpt_files

        param_dict = ms.load_checkpoint(os.path.join(temp_dir, ckpt_files[0]))
        assert param_dict


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_adaptive_checkpoint_keep_max_integration(mode):
    """
    Feature: AdaptiveModelCheckpoint with keep_checkpoint_max.
    Description: Test integration with checkpoint file management.
    Expectation: Should maintain maximum number of checkpoint files.
    """
    context.set_context(mode=mode)

    with tempfile.TemporaryDirectory() as temp_dir:
        net = SimpleNet()
        loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        optimizer = SGD(net.trainable_params(), learning_rate=0.01)

        config = AdaptiveCheckpointConfig(
            target_overhead_percentage=50.0,
            save_checkpoint_steps=2,
            keep_checkpoint_max=2
        )

        checkpoint = AdaptiveModelCheckpoint(
            prefix='keep_max_test',
            directory=temp_dir,
            config=config
        )

        model = Model(net, loss_fn, optimizer)
        dataset = create_dataset(batch_size=4, num_samples=1024)

        model.train(1, dataset, callbacks=[checkpoint])

        ckpt_files = [f for f in os.listdir(temp_dir) if f.endswith('.ckpt')]
        assert len(ckpt_files) == 2
