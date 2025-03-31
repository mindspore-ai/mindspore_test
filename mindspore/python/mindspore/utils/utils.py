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
"""mindspore utils."""
from __future__ import absolute_import

import os
from mindspore import log as logger
from mindspore import context
from mindspore import _checkparam as Validator
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._recovery_context import _set_recovery_context
from mindspore.common.api import jit_class
from mindspore._c_expression import _tft_start_record_threads, _tft_finish_record_threads


@jit_class
class ExitByRequest:
    """
    Gracefully exits the training process after get exit request.
    """

    def __init__(self):
        super(ExitByRequest, self).__init__()
        from mindspore.communication.management import get_group_size
        self.all_reduce = P.AllReduce()
        self.equal = P.Equal()
        self.assign = P.Assign()
        self.reduce_all = P.ReduceAll(keep_dims=False)
        self.group_size = get_group_size()
        self.is_distributed = self.group_size > 1
        if self.is_distributed:
            self.base = Tensor([self.group_size], dtype=mstype.int32)
        self.base1 = Tensor([1], mstype.int32)
        self.true = Tensor(True, mstype.bool_)

    def exit_by_request(self, grad, init_value, exit_value):
        """
        update GracefulExit flag by Assign op, the value is the output of AllReduce op
        :param grad: grad of net, or output of opt
        :param init_value: input value of AllReduce, a parameter
        :param exit_value: graceful exit value(out of AllReduce), update by Assign op
        :return: grad
        """
        if self.is_distributed:
            all_status = self.all_reduce(init_value)
            equal = self.equal(all_status, self.base)
            reduce_all = self.reduce_all(equal)
            grad = F.depend(grad, self.assign(exit_value, reduce_all))
        return grad


class TftHandle:
    """TftHandle class"""

    def __init__(self):
        super(TftHandle, self).__init__()
        self._controller_ip = None
        self._controller_rank_id = None
        self._controller_port = None
        self.tft = None
        self.enable_mindx = False

    def get_tft(self):
        """return tft handle"""
        return self.tft

    def unregister_tft(self):
        """unregister tft"""
        cur_rank = int(os.getenv("MS_NODE_ID"))  # from msrun
        if cur_rank == self._controller_rank_id and not self.enable_mindx:
            self.tft.tft_destroy_controller()
        self.tft.tft_destroy_processor()

    def _mindx_stub(self):
        """stub func for mindx"""
        from mindio_ttp.controller_ttp import (tft_register_mindx_callback,
                                               tft_notify_controller_stop_train,
                                               tft_notify_controller_on_global_rank,
                                               tft_notify_controller_change_strategy)

        def report_fault_ranks_func(error_rank_dict):
            tft_notify_controller_stop_train(error_rank_dict)
            return 0

        def report_stop_complete_func(code, msg, error_rank_dict):
            tft_notify_controller_on_global_rank(error_rank_dict)
            return 0

        def report_strategies_func(error_rank_dict, strategy_list):
            tft_notify_controller_change_strategy(strategy_list[-1])
            return 0

        def report_result(code, msg, error_rank_dict, curr_strategy):
            if code != 0:
                tft_notify_controller_change_strategy('dump')
            return 0

        logger.warning('Stub for mindx.')
        tft_register_mindx_callback('report_fault_ranks', report_fault_ranks_func)
        tft_register_mindx_callback('report_stop_complete', report_stop_complete_func)
        tft_register_mindx_callback('report_strategies', report_strategies_func)
        tft_register_mindx_callback('report_result', report_result)
        logger.warning('Stub register mindx func success.')

    def init(self, config=None, **kwargs):
        """
        TFT handle init fun. Mainly used to initialize the mindio component.

        Args:
            config (Dict): configs or the training model running. If ARF is not used, this parameter can be None.
            **kwargs: Reserved parameters.
        """
        tft_env = os.getenv("MS_ENABLE_TFT", "")
        if ("TTP:1" not in tft_env) and ("UCE:1" not in tft_env) and ("ARF:1" not in tft_env):
            raise ValueError("MindIO TFT regitster need custom switch on[MS_ENABLE_TFT='{TTP:1,UCE:1,ARF:1}']!")
        if "ARF:1" in tft_env:
            logger.warning(f"Disable hccl watchdog when using ARF.")
            context.set_context(ascend_config={"hccl_watchdog": False})
            logger.warning(f"Turn on TTP config when using ARF.")
            if "TTP:1,UCE:1,ARF:1" not in tft_env:
                os.environ["MS_ENABLE_TFT"] = "{TTP:1,ARF:1}"
            os.environ["MS_ENABLE_RECOVERY"] = "1"

        mode = context.get_context("mode")
        device_target = context.get_context("device_target")
        if device_target != "Ascend" or mode != context.GRAPH_MODE:
            logger.warning(f"MindIO adataper only support on Ascend device with GRAPH Mode!"
                           f"device:{device_target}, run mode: {mode}")
            return

        ctrl_port = int(os.getenv("MS_TFT_PORT"))
        ctrl_ip = os.getenv("MS_TFT_IP", "")
        Validator.check_non_negative_int(ctrl_port)
        self._controller_ip = ctrl_ip
        self._controller_rank_id = 0
        self._controller_port = ctrl_port
        try:
            from mindio_ttp import framework_ttp as tft
            self.tft = tft
        except BaseException as e:
            raise ModuleNotFoundError(f"Module nopt found. Detail info {str(e)}")
        world_size = int(os.getenv("MS_WORKER_NUM"))  # from msrun
        cur_rank = int(os.getenv("MS_NODE_ID"))  # from msrun
        enable_local_copy = False
        enable_arf = True if "ARF:1" in tft_env else False  # pylint: disable=simplifiable-if-expression
        enable_tls = False
        tls_key_dir = ""
        self.enable_mindx = os.getenv("MINDX_TASK_ID")
        # enable mindx, no need create controller
        if cur_rank == self._controller_rank_id and self.enable_mindx is None:
            logger.info(f"Begin to start tft controller on rank_id:{cur_rank}")
            if enable_arf:
                self._mindx_stub()
            self.tft.tft_init_controller(cur_rank, world_size, enable_local_copy, enable_arf=enable_arf)
            self.tft.tft_start_controller(self._controller_ip, self._controller_port, enable_tls, tls_key_dir)
            logger.info("Finish start tft controller.")

        logger.info("Begin to start tft processor.")
        _tft_start_record_threads()
        self.tft.tft_init_processor(cur_rank, world_size, enable_local_copy, enable_tls, tls_key_dir,
                                    enable_arf=enable_arf)
        self.tft.tft_start_processor(self._controller_ip, self._controller_port)
        _tft_finish_record_threads()
        logger.info("Finished start tft processor.")
        if self.tft.tft_is_reboot_node():
            logger.warning("tft report reboot init finish ")
            tft.tft_report_error(tft.ReportState.RS_INIT_FINISH.value)
            _set_recovery_context(is_reboot_node=True)
            if config is None:
                raise ValueError(
                    f"Param 'config' can not be None when using ARF, and 'config' should be a dict type.")
            logger.warning("tft reboot node no need load checkpoint when using ARF.")
            config["arf_skip_load"] = True
            ret = tft.tft_wait_next_action()
            if ret != tft.Action.RETRY.value:
                raise RuntimeError(f"ARF init failed!")


_tft_handler = TftHandle()
