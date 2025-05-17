mindspore.dataset.config.set_video_backend
==========================================

.. py:function:: mindspore.dataset.config.set_video_backend(backend)

    设置用于解码视频的后端。

    参数：
        - **backend** (str) - 视频后端类型。可为 "CPU" 或 "Ascend" 。

    异常：
        - **TypeError** - 当 `backend` 不为str类型。
        - **ValueError** - 当 `backend` 不为 "CPU" 或 "Ascend" 。
