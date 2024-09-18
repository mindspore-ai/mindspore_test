mindspore.mint.scatter
=======================================

.. py:function:: mindspore.mint.scatter(input, dim, index, src)

    根据指定索引将 `src` 中的值更新到 `input` 中返回输出。
    对于一个3D的Tensor, 输出形式如下所示：

    .. code-block::

        output[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0

        output[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1

        output[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。 `input` 的秩必须至少为1。
        - **dim** (int) - 要进行更新操作的轴。取值范围是[-r, r)，其中r是 `input` 的秩。
        - **index** (Tensor) - 输入Tensor的索引，数据类型为int32或int64的正整数。其rank必须和 `input` 一致。取值范围是[-s, s)，这里的s是 `input` 在 `axis` 指定轴的size。
        - **src** (Tensor, float) - 指定对 `input` 进行更新操作的数据。可以为Tensor，此时其数据类型必须与输入 `input` 的数据类型相同，shape必须与 `index` 相同。
          也可以是个float类型的标量。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `index` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input` 、 `index` 和 `src` 中，任意一者的秩小于1。
        - **ValueError** - `src` 的shape和 `index` 的shape不一样。
        - **ValueError** - `src` 的秩和 `input` 的秩不一样。
        - **TypeError** - `input` 的数据类型和 `src` 的数据类型不一致。
        - **RuntimeError** - `index` 中存在负数。