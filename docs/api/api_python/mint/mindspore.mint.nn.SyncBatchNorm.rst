mindspore.mint.nn.SyncBatchNorm
=================================

.. py:class:: mindspore.mint.nn.SyncBatchNorm(num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, process_group: Optional[str] = None, dtype=None)

    在N维输入上进行跨设备同步批归一化（Batch Normalization，BN）。

    同步BN是跨设备的。BN的实现仅对每个设备中的数据进行归一化。同步BN将归一化组内的输入。描述见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和和学习参数进行训练，参数见如下公式。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **num_features** (int) - 指定输入Tensor的通道数量，输入Tensor的size为 :math:`(N, C, +)` 。
        - **eps** (float) - :math:`\epsilon` 添加到分母中的值，以确保数值稳定。默认值： ``1e-5`` 。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值： ``0.1`` 。
        - **affine** (bool) - bool类型。设置为True时， :math:`\gamma` 和 :math:`\beta` 为可学习参数。默认值： ``True`` 。
        - **track_running_stats** (bool, 可选) - bool类型。设置为 ``True`` 时，会跟踪运行时的均值和方差。当设置为 ``False`` 时，
          则不会跟踪这些统计信息。且在tran和eval模式下，该cell总是使用batch的统计信息。
        - **process_group** (str, 可选) - 统计数据的同步在每个进程组内单独进行。默认行为是全局同步。默认值： ``None`` 。
        - **dtype** (:class:`mindspore.dtype`, 可选) - Parameters的dtype。默认值： ``None`` 。

    输入：
        - **x** （Tensor） - shape为 :math:`(N, C_{in}, +)` 的Tensor。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C_{out}, +)` 。

    异常：
        - **TypeError** - `num_features` 不是int。
        - **TypeError** - `eps` 不是float。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `process_group` 中的rank ID不在[0, rank_size)范围内。

    样例：

    .. note::
        在运行以下示例之前，您需要配置通信环境变量。

        对于Ascend设备，用户需要准备rank table，设置rank_id和device_id。
        这里，示例使用msrun通过单个命令跨节点拉取多进程分布式任务行指令。
        请参阅 `Ascend教程
        <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
        了解更多详情。

        此示例应在多个设备上运行。