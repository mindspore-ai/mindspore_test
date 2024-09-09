mindspore.train.TFTRegister
================================

.. py:class:: mindspore.train.TFTRegister(ctrl_rank_id, ctrl_ip, ctrl_port, ckpt_save_path)

    该回调用于开启 `MindIO的TTP特性 <https://www.hiascend.com/document/detail/zh/mindx-dl/60rc1/mindio/mindiottp/mindiottp001.html>`_，该CallBack会嵌入训练的流程，完成TTP 的初始化、上报、异常处理等操作。

    .. note::
        该特性仅支持Ascend后端的静态图模式，并且只支持sink_size值小于等于1的场景。

    参数：
        - **ctrl_rank_id** (int) - TTP controller 运行的rank_id, 该参数用于启动TTP的controller。
        - **ctrl_ip** (str) - TTP controller 的IP地址, 该参数用于启动TTP的controller。
        - **ctrl_port** (int) - TTP controller 的IP端口, 该参数用于启动TTP的controller和processor。
        - **ckpt_save_path** (str) -  异常发生时ckpt保存的路径，该路径是一个目录，ckpt的异常保存时会在该录下创建新的名为‘ttp_saved_checkpoints-step_{cur_step_num}’目录。

    异常：
        - **Exception** - TTP 初始化失败，会对外抛Exception异常。
        - **ModuleNotFoundError** - Mindio TTP whl 包未安装。

    .. py:method:: on_train_step_end(run_context)

        每个step完成时进行MindIO TTP的上报。

        参数：
            - **run_context** (RunContext) - 包含模型的相关信息。详情请参考 :class:`mindspore.train.RunContext`。