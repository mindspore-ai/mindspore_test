mindspore.ops.Morph
==============================

.. py:class:: mindspore.ops.Morph(fn, infer_shape, infer_dtype)

    `Morph` 算子用于对用户自定义函数 `fn` 进行封装，允许其被当做自定义算子使用。

    `Morph` 算子的主要适用于静态图的分布式自动并行场景，通过在自定义函数 `fn` 中使用集合通信算子，实现自定义的并行计算逻辑，尤其适用于 `fn` 内存在动态Shape的场景。

    `Morph` 算子作用于输入时，实际上是其内封装的自定义函数 `fn` 作用于输入。

    `Morph` 算子与 :func:`mindspore.ops.Custom` 的主要区别在于，前者会在自动微分前被展开替换为用户自定义 `fn`，故无需实现反向函数。

    .. note::
        - 本算子只支持图模式。
        - `fn` 必须满足图模式语法约束。
        - 用户无需实现自定义反向函数。

    参数：
        - **fn** (Function) - MindSpore Function，用户自定义函数。
        - **infer_shape** (Function) - Mindspore Function，用户自定义 `infer_shape` 函数。
        - **infer_dtype** (Function) - Mindspore Function，用户自定义 `infer_dtype` 函数。

    输入：
        用户自定义 `fn` 的输入。

    输出：
        用户自定义 `fn` 的输出。

    异常：
        - **RuntimeError** - 如果算子在非图模式下被使用。
