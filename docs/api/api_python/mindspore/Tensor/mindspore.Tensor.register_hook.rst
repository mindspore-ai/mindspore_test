mindspore.Tensor.register_hook
==============================

.. py:method:: mindspore.Tensor.register_hook(hook_fn)

    设置Tensor对象的反向hook函数。

    .. note::
        - `hook_fn` 必须有如下代码定义： `grad` 是反向传递给 `Tensor` 对象的梯度。 用户可以在 `hook_fn` 中打印梯度数据或者返回新的输出梯度。
        - `hook_fn` 返回新的梯度输出，不能不设置返回值： `hook_fn(grad) -> New grad_output`。
        - 静态图模式下需满足如下约束：

          - `hook_fn` 同样需满足静态图模式下的语法约束。
          - 不支持在图内（即 `Cell.construct` 函数或被 `@jit` 修饰的函数）对 `Parameter` 注册 `hook_fn`。
          - 不支持在图内对 `hook_fn` 进行删除。
          - 图内对 `Tensor` 注册 `hook_fn` 将返回 `Tensor` 本身。

    参数：
        - **hook_fn** (function) - 捕获 `Tensor` 反向传播时的梯度，并输出或更改该梯度的 `hook_fn` 函数。

    返回：
        返回与该 `hook_fn` 函数对应的 `handle` 对象。可通过调用 `handle.remove()` 来删除添加的 `hook_fn` 函数。

    异常：
        - **TypeError** - 如果 `hook_fn` 不是Python函数。
