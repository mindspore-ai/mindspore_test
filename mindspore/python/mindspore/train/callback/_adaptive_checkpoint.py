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
"""Adaptive checkpoint related classes and functions."""
from __future__ import absolute_import

import time
import math

from mindspore import log as logger
from mindspore.train.callback._checkpoint import CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import get_group_size, GlobalComm
from mindspore.common.tensor import Tensor
from mindspore.parallel._cell_wrapper import SingleCommunicator
from mindspore.common import dtype as mstype


class AdaptiveCheckpointConfig(CheckpointConfig):
    """
    The adaptive configuration of model checkpoint.

    This class extends CheckpointConfig to support adaptive checkpoint saving based on
    target overhead percentage or failure rate.

    Note:
        Only one of `target_overhead_percentage` or `failure_rate` can be set; both cannot be configured simultaneously.

    Args:
        target_overhead_percentage (float, optional): Target percentage of training time that
            checkpoint saving should consume. When set, the checkpoint saving interval will be
            dynamically adjusted to maintain this overhead percentage. Default: ``None``.
        failure_rate (float, optional): Expected failure rate for optimal checkpoint interval
            calculation. When set, uses the formula: interval = sqrt(2 * checkpoint_time /
            (failure_rate * step_time)) to determine optimal saving frequency. Default: ``None``.
        **kwargs: Other arguments passed to CheckpointConfig.

    Raises:
        ValueError: If both `target_overhead_percentage` and `failure_rate` are set.
        ValueError: If `target_overhead_percentage` is not in valid range (0, 100].
        ValueError: If `failure_rate` is not in valid range (0, 1].

    Examples:
        >>> from mindspore.train.callback import AdaptiveCheckpointConfig, AdaptiveModelCheckpoint
        >>>
        >>> # Configure adaptive saving based on overhead percentage
        >>> config = AdaptiveCheckpointConfig(
        ...     target_overhead_percentage=5.0,  # 5% of training time for checkpointing
        ... )
        >>>
        >>> # Configure adaptive saving based on failure rate
        >>> config = AdaptiveCheckpointConfig(
        ...     failure_rate=0.001,  # Expected failure rate of 0.1%
        ... )
    """

    def __init__(self,
                 target_overhead_percentage=None,
                 failure_rate=None,
                 **kwargs):
        if target_overhead_percentage is not None and failure_rate is not None:
            raise ValueError("For 'AdaptiveCheckpointConfig', only one of 'target_overhead_percentage' "
                             "or 'failure_rate' can be set, not both.")

        if target_overhead_percentage is not None:
            if not isinstance(target_overhead_percentage,
                              (int, float)) or target_overhead_percentage <= 0 or target_overhead_percentage > 100:
                raise ValueError("For 'AdaptiveCheckpointConfig', 'target_overhead_percentage' must be "
                                 "a number in range (0, 100].")

        if failure_rate is not None:
            if not isinstance(failure_rate, (int, float)) or failure_rate <= 0 or failure_rate > 1:
                raise ValueError("For 'AdaptiveCheckpointConfig', 'failure_rate' must be "
                                 "a number in range (0, 1].")

        super(AdaptiveCheckpointConfig, self).__init__(**kwargs)

        self.target_overhead_percentage = target_overhead_percentage
        self.failure_rate = failure_rate

    @property
    def is_adaptive(self):
        """
        Check if adaptive checkpoint saving is enabled.

        Returns:
            bool, whether adaptive checkpoint saving is enabled.
        """
        return self.target_overhead_percentage is not None or self.failure_rate is not None

    @property
    def adaptive_mode(self):
        """
        Get the adaptive mode.

        Returns:
            str. The adaptive mode, which can be ``'percentage'`` or ``'failure_rate'``;
            returns ``None`` if adaptive mode is not enabled.
        """
        if self.target_overhead_percentage is not None:
            return "percentage"
        if self.failure_rate is not None:
            return "failure_rate"
        return None


class AdaptiveModelCheckpoint(ModelCheckpoint):
    """
    The adaptive checkpoint callback class.

    This class extends ModelCheckpoint to support adaptive checkpoint saving that dynamically
    adjusts the saving interval based on training performance metrics.

    Args:
        prefix (Union[str, callable object], optional): The prefix name or callable object to
            generate name of checkpoint files. Default: ``'CKP'``.
        directory (Union[str, callable object], optional): The folder path where the checkpoint is stored,
            or the callable object used to generate the path. By default, the file is saved in the current directory.
            Default: ``None``.
        config (AdaptiveCheckpointConfig, optional): Adaptive checkpoint strategy configuration. Default: ``None``.

    Raises:
        TypeError: If the config is not AdaptiveCheckpointConfig type.

    Examples:
        >>> from mindspore.train.callback import AdaptiveCheckpointConfig, AdaptiveModelCheckpoint
        >>>
        >>> # Create adaptive checkpoint configuration
        >>> config = AdaptiveCheckpointConfig(
        ...     target_overhead_percentage=5,
        ... )
        >>>
        >>> # Create adaptive checkpoint callback
        >>> adaptive_ckpt = AdaptiveModelCheckpoint(
        ...     prefix='adaptive_model',
        ...     directory='./checkpoints',
        ...     config=config
        ... )
    """

    def __init__(self, prefix='CKP', directory=None, config=None):
        if config is not None and not isinstance(config, AdaptiveCheckpointConfig):
            raise TypeError("For 'AdaptiveModelCheckpoint', the type of argument 'config' should be "
                            "'AdaptiveCheckpointConfig', but got {}.".format(type(config)))

        super(AdaptiveModelCheckpoint, self).__init__(prefix, directory, config)

        self._adaptive_config = config if config is not None else AdaptiveCheckpointConfig(target_overhead_percentage=1)

        if self._adaptive_config.is_adaptive:
            if self._adaptive_config.save_checkpoint_steps is None or self._adaptive_config.save_checkpoint_steps <= 0:
                raise ValueError("For adaptive checkpoint saving, 'save_checkpoint_steps' must be configured "
                                 "and greater than 0.")

            self.interval_comm = SingleCommunicator(GlobalComm.WORLD_COMM_GROUP)
            self.group_size = get_group_size()

            self._step_times = []
            self._checkpoint_times = []
            self._current_save_interval = self._adaptive_config.save_checkpoint_steps
            self._adaptive_phase = self._adaptive_config.adaptive_mode

            self._step_start_time = None
            self._checkpoint_start_time = None
        else:
            self._adaptive_phase = None

    def step_begin(self, run_context):
        """
        Record step start time for adaptive timing.

        Args:
            run_context (RunContext): Contains some basic information about the model.
                For details, please refer to :class:`mindspore.train.RunContext` .
        """
        super(AdaptiveModelCheckpoint, self).step_begin(run_context)
        if self._adaptive_phase is not None:
            self._step_start_time = time.time()

    def step_end(self, run_context):
        """
        Save checkpoint and update adaptive intervals.

        Args:
            run_context (RunContext): Contains some basic information about the model.
                For details, please refer to :class:`mindspore.train.RunContext` .
        """
        if self._adaptive_phase is not None:
            if self._step_start_time is not None:
                step_time = time.time() - self._step_start_time
                self._step_times.append(step_time)

        super(AdaptiveModelCheckpoint, self).step_end(run_context)

    def _check_save_ckpt(self, cb_params, force_to_save):
        """Check whether save checkpoint files or not with adaptive interval."""
        if self._adaptive_phase is not None and self._current_save_interval > 0:
            if cb_params.cur_step_num >= self._last_triggered_step + self._current_save_interval \
                    or force_to_save is True:
                return True
        else:
            return super(AdaptiveModelCheckpoint, self)._check_save_ckpt(cb_params, force_to_save)
        return False

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint with adaptive timing measurement."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return

        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)

        if save_ckpt and self._adaptive_phase is not None:
            self._checkpoint_start_time = time.time()

        super(AdaptiveModelCheckpoint, self)._save_ckpt(cb_params, force_to_save)

        if save_ckpt and self._adaptive_phase is not None:
            checkpoint_time = time.time() - self._checkpoint_start_time
            self._checkpoint_times.append(checkpoint_time)
            self._update_save_interval()

    def _update_save_interval(self):
        """Update save interval based on current adaptive phase and measurements."""
        if self._adaptive_phase == "percentage":
            self._update_percentage_phase_interval()
        elif self._adaptive_phase == "failure_rate":
            self._update_failure_rate_phase_interval()

    def _update_percentage_phase_interval(self):
        """Update save interval during percentage-based adaptive phase."""
        if len(self._step_times) <= 1:
            return

        avg_checkpoint_time = self._get_avg_checkpoint_time()
        avg_step_time = self._get_avg_step_time()

        if avg_checkpoint_time is None or avg_step_time is None:
            return

        scaled_checkpoint_time = 100 * avg_checkpoint_time / self._adaptive_config.target_overhead_percentage
        numerator = scaled_checkpoint_time - avg_checkpoint_time
        new_interval = numerator / avg_step_time
        new_interval = self._comm_interval_step(new_interval)

        if new_interval != self._current_save_interval:
            logger.info(f"According to percentage of time consumed by checkpoint saving, "
                        f"adjust save interval: {self._current_save_interval} -> {new_interval}. "
                        f"(checkpoint_time: {avg_checkpoint_time:.6f}s, step_time: {avg_step_time:.6f}s, "
                        f"percentage: {self._adaptive_config.target_overhead_percentage}%)")
            self._current_save_interval = new_interval

    def _update_failure_rate_phase_interval(self):
        """Update save interval during failure rate-based adaptive phase."""
        if len(self._step_times) <= 1:
            return

        avg_checkpoint_time = self._get_avg_checkpoint_time()
        avg_step_time = self._get_avg_step_time()

        if avg_checkpoint_time is None or avg_step_time is None:
            return

        optimal_interval = math.sqrt(2 * avg_checkpoint_time / (self._adaptive_config.failure_rate * avg_step_time))
        new_interval = self._comm_interval_step(optimal_interval)

        if new_interval != self._current_save_interval:
            logger.info(f"According to failure rate, adjust save interval: {self._current_save_interval} "
                        f"-> {new_interval}. (checkpoint_time: {avg_checkpoint_time:.6f}s, "
                        f"step_time: {avg_step_time:.6f}s, failure_rate: {self._adaptive_config.failure_rate})")
            self._current_save_interval = new_interval

    def _get_avg_checkpoint_time(self):
        """Get the average checkpoint time."""
        if self._checkpoint_times:
            return sum(self._checkpoint_times) / len(self._checkpoint_times)
        return None

    def _get_avg_step_time(self, interval_step=2000):
        """Get the average step time."""
        if len(self._step_times) <= 1:
            return None

        if len(self._step_times) <= interval_step:
            return sum(self._step_times[1:]) / len(self._step_times[1:])
        return sum(self._step_times[-interval_step:]) / len(self._step_times[-interval_step:])

    def _comm_interval_step(self, interval_step):
        """Get the average interval step according to the group size."""
        if self.group_size == 1:
            return max(1, int(round(interval_step)))

        new_interval = self.interval_comm(Tensor(int(round(interval_step)), dtype=mstype.int32))
        return max(1, int(new_interval.asnumpy().item() / self.group_size))
