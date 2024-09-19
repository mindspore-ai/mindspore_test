mindspore.numpy.piecewise
=================================

.. py:function:: mindspore.numpy.piecewise(x, condlist, funclist, *args, **kw)

    执行分段定义的函数。根据一组条件和相应的函数，在条件为真的位置对输入数据进行函数求值。

    参数：
        - **x** (Union[int, float, bool, list, tuple, Tensor]) - 输入定义域。
        - **condlist** (Union[bool, list[Tensor, bool]]) - 每个boolean数组与 ``funclist`` 中的一个函数相对应。在 ``condlist[i]`` 为真时，使用 ``funclist[i](x)`` 作为输出值。 ``condlist`` 中的每个boolean数组选择 ``x`` 的一个部分，因此应该与 ``x`` 的shape相同。 ``condlist`` 的长度必须与 ``funclist`` 的长度相对应。如果给出了一个额外的函数，即 :math:`len(funclist) == len(condlist) + 1` ，则该额外的函数为默认值，用于所有条件均为假的地方。
        - **funclist** (Union[list[callables], list[scalars]]) - 每个函数在其相应的条件为真时对 ``x`` 进行计算。它应该接受一个一维数组作为输入，并输出一个一维数组或标量值。如果提供的不是可调用对象，而是标量，则假定为常数函数 ``(lambda x: scalar)`` 。
        - **args** (any) - 传递给 ``piecewise`` 的任何进一步参数都会在执行时传递给函数，例如，如果调用 ``piecewise(..., ..., 1, 'a')`` ，则每个函数都将被调用为 ``f(x, 1, 'a')`` 。
        - **kw** (any) - 用于调用 ``piecewise`` 的关键字参数，在执行时传递给函数，例如，如果调用 ``piecewise(..., ..., alpha=1)`` ，则每个函数将被调用为 ``f(x, alpha=1)`` 。

    返回：
        Tensor，输出与 ``x`` 具有相同的shape和类型，通过对 ``x`` 的对应部分调用 ``funclist`` 中的函数找到输出值，这些部分由 ``condlist`` 中的boolean数组定义。未被任何条件覆盖的部分默认值为 ``0`` 。

    异常：
        - **ValueError** - 如果 ``funclist`` 的长度不在 :math:`(len(condlist), len(condlist) + 1)` 范围内。