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
"""Checkpoint related classes and functions."""

import os
from mindspore.train.serialization import save_checkpoint
from mindspore.parallel._utils import _get_device_num
from mindspore import _checkparam as Validator
from mindspore.train.callback._callback import Callback
from mindspore import context
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.communication import get_rank, get_group_size
from mindspore import log as logger
from mindspore.train.serialization import _get_cur_rank_dp
from mindspore._c_expression import _repair_device, _stop_device, _tft_sem_post
from mindspore._c_expression import clean_tdt_channel
from mindspore._c_expression import send_recv
from mindspore._c_expression import CollectiveManager
from mindspore._c_expression import _get_uce_process_strategy, _get_uce_mem_info
from mindspore._c_expression import Tensor as Tensor_
import mindspore
import mindspore.common.dtype as mstype

def _get_ckpt_dir(step, ckpt_save_path, is_tmp_file):
    """ Common func to generate ckpt dir name."""
    tmp = "_tmp" if is_tmp_file else ""
    mid_dir = f"tft_saved_checkpoints-step_{str(step)}{tmp}"
    return os.path.join(ckpt_save_path, mid_dir)

def _save_checkpoint_on_failure(step, save_info, args, cb_ctx):
    """ Callback used for TFT save ckpt function when errors occur."""
    logger.info("Enter _save_checkpoint_on_failure function")
    if not cb_ctx._is_params_consistent():    # pylint: disable=W0212
        raise RuntimeError("Can't save parameters, because they are left in inconsistent state!")

    ckpt_save_path = cb_ctx.ckpt_save_path
    cb_params = args
    cur_rank = get_rank()
    cur_step_num = cb_params.cur_step_num
    cur_epoch_num = cb_params.cur_epoch_num
    batch_num = cb_params.batch_num
    if cur_step_num > step:
        cur_epoch_num = (step - 1) // batch_num + 1
    step_num_in_epoch = int((step - 1) % batch_num + 1)

    append_dict = {}
    append_dict["epoch_num"] = cur_epoch_num
    append_dict["step_num"] = step
    append_dict["cur_rank"] = cur_rank
    append_dict["batch_num"] = batch_num
    append_dict["__exception_save__"] = True

    append_dict["global_step"] = Parameter([cb_ctx.global_step])
    outputs = cb_params.net_outputs
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
        append_dict["loss_scale"] = outputs[2]

    ckpt_file = f"ttp_rank_{str(cur_rank)}-{str(cur_epoch_num)}_{str(step_num_in_epoch)}.ckpt"
    cur_ckpt_dir = _get_ckpt_dir(step, ckpt_save_path, True) + "/rank_" + str(cur_rank)
    os.makedirs(cur_ckpt_dir, exist_ok=True)
    cur_file = os.path.join(cur_ckpt_dir, ckpt_file)
    save_checkpoint(cb_params.train_network, cur_file,
                    integrated_save=False, append_dict=append_dict)
    logger.info("Finish _save_checkpoint_on_failure function")

def _rename_save_result(step, cb_ctx):
    """ Callback used for TFT rename function after ckpt save callback was finished and successful."""
    logger.info("Enter _rename_save_result function")
    tmp_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, True)
    fin_dir = _get_ckpt_dir(step, cb_ctx.ckpt_save_path, False)

    os.rename(tmp_dir, fin_dir)
    logger.info("Finish _rename_save_result function")

def _tft_exit_cb(ctx):
    logger.error("Enter mindio ttp exit process, which means other ranks occur exception, check other ranks' logs!")
    _tft_sem_post()
    os._exit(1)   # pylint: disable=W0212

def _tft_repair_callback(step, need_rebuild, error_ranks, repair_info, args, cb_ctx):
    """ Callback used for TFT repair function."""
    logger.info("Enter _tft_repair_callback repair type: {}".format(repair_info["repair_type"]))
    if(repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value\
or repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_LOWLEVEL.value):
        logger.info("Enter _tft_repair_callback uce REPARI_DEVICE device_id : {}".format(cb_ctx.device_id))
        _repair_device(cb_ctx.device_id)

    if(repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_UCE_HIGHLEVEL.value\
       or repair_info["repair_type"] == cb_ctx.tft.RepairType.RT_SEND.value):
        logger.info("Enter _tft_repair_callback SEND_RECV repair type: \
{}, src_rank:{}, dst_rank: {}".format(repair_info["repair_type"], repair_info["src"], repair_info["dst"]))
        cb_params = args
        src_rank = repair_info["src"][0]
        dst_rank = repair_info["dst"][0]
        send_recv(cb_params.network.trainable_params(), src_rank, dst_rank)
    logger.info("Finish _tft_repair_callback")


def _tft_clean_callback(is_uce_error, ctx):
    """ Callback used for TFT clean function."""
    logger.info("Enter _tft_clean_callback")
    ret = 0
    if is_uce_error:
        _get_uce_mem_info(ctx.device_id)
        err_strategy = _get_uce_process_strategy()
        logger.info("_tft_clean_callback err_strategy: {}".format(err_strategy))
        if err_strategy == "RS_UCE_HIGHLEVEL":
            ret = 0
        elif err_strategy == "RS_UCE_LOWLEVEL":
            ret = 2
        else:
            ret = 1
    clean_tdt_channel()
    logger.info("Enter _tft_clean_callback resume_hccl_comm")
    CollectiveManager.get_instance().resume_hccl_comm()
    logger.info("Finish _tft_clean_callback, ret: {}".format(ret))
    return ret


def _tft_stop_callback(cb_ctx):
    """ Callback used for TFT stop function."""
    logger.info("Enter _tft_stop_callback device_id: {}".format(cb_ctx.device_id))
    _stop_device(cb_ctx.device_id)
    if not cb_ctx._is_params_consistent():    # pylint: disable=W0212
        raise RuntimeError("Can't stop device, because training parameters are left in inconsistent state!")
    logger.info("Finish _tft_stop_callback")


class TFTRegister(Callback):
    """
    This callback is used to enable the TFT feature
    `MindIO TFT <https://www.hiascend.com/document/detail/zh/mindx-dl/60rc2/mindio/mindiottp/mindiottp001.html>`_.
    This callback will execute TFT operations during training process, such as TFT init, report and exception handle.

    Note:
        Required for Ascend graph mode only. And sink size must be less than or equal to 1.

    Args:
        ctrl_rank_id (int): TFT controller's running rank_id, used for init TFT controller.
        ctrl_ip (str): TFT controller's ip address, used for init TFT controller.
        ctrl_port (int): TFT controller's ip port, used for init TFT controller and processor.
        ckpt_save_path (str): Checkpoint save directory when failure occurs, checkpoint file will save to directory
           named ttp_saved_checkpoints-step_{cur_step_num} under this directory.

    Raises:
        Exception: TFT init failed.
        ModuleNotFoundError: Mindio TFT whl package is not installed.

    Examples:
        >>> import numpy as np
        >>> import os
        >>> import math
        >>> import mindspore as ms
        >>> import mindspore.dataset as ds
        >>> from mindspore import nn, ops, Parameter, train
        >>> from mindspore.communication import init
        >>> from mindspore.common.initializer import initializer, HeUniform
        >>> from mindspore.train import Model, TFTRegister
        >>> from mindspore import dataset as ds
        >>> ms.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
        >>> ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
        >>> init()
        >>> ms.set_seed(1)
        >>> ms.set_auto_parallel_context(strategy_ckpt_config={"save_file":
        >>>                             "./src_pipeline_strategys/src_strategy_{}.ckpt".format(get_rank())})
        >>> class MatMulCell(nn.Cell):
        ...     def __init__(self, param=None, shape=None):
        ...         super().__init__()
        ...         if shape is None:
        ...             shape = [28 * 28, 512]
        ...         weight_init = HeUniform(math.sqrt(5))
        ...         self.param = Parameter(initializer(weight_init, shape), name="param")
        ...         if param is not None:
        ...             self.param = param
        ...         self.print = ops.Print()
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         out = self.matmul(x, self.param)
        ...         self.print("out is:", out)
        ...         return out
        >>>
        >>> class Network(nn.Cell):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.flatten = nn.Flatten()
        ...         self.layer1 = MatMulCell()
        ...         self.relu1 = nn.ReLU()
        ...         self.layer2 = nn.Dense(512, 512)
        ...         self.relu2 = nn.ReLU()
        ...         self.layer3 = nn.Dense(512, 10)
        ...
        ...     def construct(self, x):
        ...         x = self.flatten(x)
        ...         x = self.layer1(x)
        ...         x = self.relu1(x)
        ...         x = self.layer2(x)
        ...         x = self.relu2(x)
        ...         logits = self.layer3(x)
        ...         return logits
        >>>
        >>> net = Network()
        >>> net.layer1.pipeline_stage = 0
        >>> net.relu1.pipeline_stage = 0
        >>> net.layer2.pipeline_stage = 0
        >>> net.relu2.pipeline_stage = 1
        >>> net.layer3.pipeline_stage = 1
        >>>
        >>> def create_dataset(batch_size):
        ...     dataset_path = os.getenv("DATA_PATH")
        ...     dataset = ds.MnistDataset(dataset_path)
        ...     image_transforms = [
        ...         ds.vision.Rescale(1.0 / 255.0, 0),
        ...         ds.vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        ...         ds.vision.HWC2CHW()
        ...     ]
        ...     label_transform = ds.transforms.TypeCast(ms.int32)
        ...     dataset = dataset.map(image_transforms, 'image')
        ...     dataset = dataset.map(label_transform, 'label')
        ...     dataset = dataset.batch(batch_size)
        ...     return dataset
        >>>
        >>> data_set = create_dataset(32)
        >>>
        >>> optimizer = nn.SGD(net.trainable_params(), 1e-2)
        >>> optimizer_wrapper = nn.OptTFTWrapper(optimizer)
        >>> loss_fn = nn.CrossEntropyLoss()
        >>>
        >>> net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
        >>> net_with_loss.set_train()
        >>> model = Model(net_with_loss, optimizer=optimizer)
        >>> tft_cb = TFTRegister("192.168.0.1", 2000, "./tft_checkpoint/")
        >>> loss_cb = train.LossMonitor(1)
        >>> model.train(1, dataset, callbacks=[tft_cb, loss_cb])
    """

    def __init__(self, ctrl_rank_id, ctrl_ip, ctrl_port, ckpt_save_path):
        super(TFTRegister, self).__init__()

        tft_env = os.getenv("MS_ENABLE_TFT", "")
        if ("TTP:1" not in tft_env) and ("UCE:1" not in tft_env):
            raise ValueError("MindIO TFT regitster need custom switch on[MS_ENABLE_TFT='{TTP:1,UCE:1}']!")
        mode = context.get_context("mode")
        device_target = context.get_context("device_target")
        if device_target != "Ascend" or mode != context.GRAPH_MODE:
            raise ValueError("MindIO adataper only support on Ascend device with GRAPH Mode!")

        # let it raise errors if not install mindio_tft package
        from mindio_ttp import framework_ttp as tft
        self.tft = tft
        self.global_step = 0
        Validator.check_non_negative_int(ctrl_port)
        self.has_init_replica = False
        self._controller_ip = ctrl_ip
        self._controller_rank_id = ctrl_rank_id
        self._controller_port = ctrl_port
        self.cb_params = None
        self.device_id = context.get_context("device_id")
        self._init_tft()
        self.ckpt_save_path = ckpt_save_path
        self.assign = mindspore.ops.Assign()
        self.g_one = Parameter(Tensor([1], dtype=mstype.int32))
        self.s1 = mindspore.hal.Stream()

    def _is_params_consistent(self):
        for key, param in self.cb_params.train_network.parameters_and_names():
            if "tft_g_one_flag" in key:
                with mindspore.hal.StreamCtx(self.s1):
                    tft_g_one_flag = Tensor(Tensor_.move_to(param, "CPU", False))
                self.s1.synchronize()
                return int(tft_g_one_flag) == 1
        return False

    def _set_tft_optimizer_replica(self, run_context):
        """ set Mindio TFT optimizer replica info, used internal. """
        cur_rank = get_rank()
        cb_params = run_context.original_args()
        train_network = cb_params.train_network
        # in data_parallel mode, every ranks has same train parameters
        if context.get_auto_parallel_context("parallel_mode") == "data_parallel":
            group_size = get_group_size()
            dp = tuple(range(group_size))
        else:
            param_layout_dict = train_network.parameter_layout_dict
            dp = _get_cur_rank_dp(param_layout_dict) if param_layout_dict else _get_cur_rank_dp(train_network)
        logger.warning(f"Set TFT replica with dp: {dp}.")
        replica_info = [
            {
                "type": 1,
                "rank_list": dp,
                "replica_cnt": len(dp),
                "replica_shift": 0
            }
        ]
        self.tft.tft_set_optimizer_replica(cur_rank, replica_info)

    def _init_tft(self):
        """ Init Mindio TFT, used internal. """
        logger.info("Begin to init tft.")
        self.tft.tft_register_save_ckpt_handler(_save_checkpoint_on_failure, self)
        self.tft.tft_register_rename_handler(_rename_save_result, self)
        self.tft.tft_register_exit_handler(_tft_exit_cb, self)
        self.tft.tft_register_stop_handler(_tft_stop_callback, self)
        self.tft.tft_register_clean_handler(_tft_clean_callback, self)
        self.tft.tft_register_repair_handler(_tft_repair_callback, self)

        world_size = _get_device_num()
        cur_rank = get_rank()
        enable_local_copy = False
        enable_arf = False
        enable_zit = False
        enable_tls = False
        tls_key_dir = ""

        if cur_rank == self._controller_rank_id:
            logger.info(f"Begin to start tft controller on rank_id:{cur_rank}")
            self.tft.tft_init_controller(cur_rank, world_size, enable_local_copy, enable_arf, enable_zit)
            self.tft.tft_start_controller(self._controller_ip, self._controller_port, enable_tls, tls_key_dir)
            logger.info("Finish start tft controller.")

        logger.info("Begin to start tft processor.")
        self.tft.tft_init_processor(cur_rank, world_size, enable_local_copy, enable_tls, tls_key_dir)
        self.tft.tft_start_processor(self._controller_ip, self._controller_port)
        logger.info("Finished start tft processor.")

    def on_train_step_end(self, run_context):
        """
        And report status to MindIO TFT after every step finished.

        Args:
            run_context (RunContext): Context of the train running. Refer to
                                      :class:`mindspore.train.RunContext` for detail.
        """
        if self.has_init_replica is False:
            self.has_init_replica = True
            self._set_tft_optimizer_replica(run_context)
        cb_params = run_context.original_args()
        logger.info("START Set optimizer finish step status to TFT. step: {}".format(cb_params.cur_step_num))
        self.tft.tft_end_updating_os(cb_params.cur_step_num)
        if cb_params.optimizer is not None:
            self.global_step = int(cb_params.optimizer.global_step.data)
            self.assign(cb_params.optimizer.tft_g_one_flag, self.g_one)
        else:
            self.global_step = int(cb_params.network.optimizer.global_step.data)
            self.assign(cb_params.network.optimizer.tft_g_one_flag, self.g_one)
        logger.info("END Set optimizer finish step status to TFT.")


    def on_train_begin(self, run_context):
        cb_params = run_context.original_args()
        sink_size = cb_params.get("sink_size", 0)
        if sink_size > 1:
            raise ValueError("TFT feature doesn't support sink_size > 1.")
        logger.info("Set set args to TFT.")
        self.tft.tft_set_step_args(cb_params)
        self.cb_params = cb_params

    def end(self, run_context):
        cur_rank = get_rank()
        if cur_rank == self._controller_rank_id:
            self.tft.tft_destroy_controller()
        self.tft.tft_destroy_processor()
