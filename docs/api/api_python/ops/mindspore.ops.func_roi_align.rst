mindspore.ops.roi_align
=======================

.. py:function:: mindspore.ops.roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)

    感兴趣区域对齐(RoI Align)运算。

    RoI Align通过在特征图上对附近网格点进行双线性插值计算每个采样点。参阅论文 `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor[N, C, H, W]) - 输入特征，shape :math:`(N, C, H, W)` 。数据类型支持float32。
        - **boxes** (Tensor[K, 5]) - shape :math:`(K, 5)` 。数据类型支持float32。 `K` 为RoI的数量。第二个维度的大小必须为 `5` ，分别代表 :math:`(image\_index, x1, y1, x2, y2)` 。
          `image_index` 表示图像的索引； `x1` 和 `y1` 分别对应RoI左上角坐标值； `x2` 和 `y2` 分别对应RoI右下角坐标值。坐标需要满足 `0 <= x1 < x2` 且 `0 <= y1 < y2` 。
        - **output_size** (Union[int, tuple(int)]) - 执行池化后的输出大小，:math:`(pooled\_height, pooled\_width)`。
        - **spatial_scale** (float，可选) - 缩放系数。将原始图像坐标映射到输入特征图坐标。默认值： ``1.0`` 。设RoI的高度在原始图像中为 `ori_h` ，在输入特征图中为 `fea_h` ，则 `spatial_scale` 应为 `fea_h / ori_h` 。
        - **sampling_ratio** (int，可选) - 采样数。默认值： ``-1`` 。如果大于0，使用 :math:`sampling\_ratio x sampling\_ratio` ；
          如果小于等于0，使用 :math:`ceil(roi\_height / output\_height) x ceil(roi\_width / output\_width)` 。
        - **aligned** (bool，可选) - 如果值为 ``False`` ，则使用该算子的历史实现。如果值为 ``True`` ，则像素偏移框会将其坐标调整-0.5，以便与两个相邻的像素索引更好地对齐。默认值： ``False`` 。

    返回：
        Tensor，shape :math:`(K, C, output\_size[0], output\_size[1])` 。

    异常：
        - **TypeError** - `sampling_ratio` 不是int类型。
        - **TypeError** - `spatial_scale` 不是float类型。
        - **TypeError** - `input` 或 `boxes` 不是Tensor。
        - **TypeError** - `aligned` 不是bool类型。
