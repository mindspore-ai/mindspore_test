import collections
import contextlib
import copy
import re
import numpy as np
from typing import Callable, Optional, Union

import mindspore as ms
from mindspore import ops

# S：字节字符串类型（bytes）。
# a：字节字符串类型的旧版本别名（与 S 相同）。
# U：Unicode 字符串类型。
# O：Python object类型。
np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`mindspore.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`mindspore.Tensor`.
    If the input is not an NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~mindspore.dataset.dataloader.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~mindspore.dataset.dataloader.DataLoader.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted
    """
    elem_type = type(data)

    # return if tensor
    if isinstance(data, ms.Tensor):
        return data
    
    # only convert numeric numpy, ignore str/obj numpy
    if isinstance(data, np.ndarray):
        if np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return ms.Tensor.from_numpy(data)

    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            else:
                return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `__copy__` / `update(mapping)`
            # or `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, d in enumerate(data):
                    clone[i] = default_convert(d)
                return clone
            else:
                return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `__copy__` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data
    

def collate(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    r"""
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # Extend this function to handle batch of tensors
        ...     return mindspore.ops.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {mindspore.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({mindspore.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    #import pdb;pdb.set_trace()
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)
    
    # for those types not in collate_fn_map
    if isinstance(elem, collections.abc.Mapping):
        try:
            if isinstance(elem, collections.abc.MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate(batch):
    r"""
    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.

    The exact output type can be a :class:`mindspore.Tensor`, a `Sequence` of :class:`mindspore.Tensor`, a
    Collection of :class:`mindspore.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~mindspore.dataset.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`mindspore.Tensor` -> :class:`mindspore.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`mindspore.Tensor`
        * `float` -> :class:`mindspore.Tensor`
        * `int` -> :class:`mindspore.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated
    """
    def collate_tensor_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        return ops.stack(batch, axis=0)
        #return ms.mint.stack(batch, dim=0)
    
    def collate_numpy_array_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        elem = batch[0]
        # array of string classes and object
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            raise TypeError(default_collate_err_msg_format.format(elem.dtype))
        return collate([ms.Tensor.from_numpy(b) for b in batch], collate_fn_map=collate_fn_map)

    def collate_numpy_scalar_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        return ms.Tensor.from_numpy(batch)

    def collate_float_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        return ms.Tensor(batch, dtype=ms.float64)


    def collate_int_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        return ms.Tensor(batch)


    def collate_str_fn(
        batch,
        *,
        collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
    ):
        return batch  #ms.tensor(batch)

    default_collate_fn_map: dict[Union[type, tuple[type, ...]], Callable] = {
        ms.Tensor: collate_tensor_fn,
        np.ndarray: collate_numpy_array_fn,
        (np.bool_, np.number, np.object_): collate_numpy_scalar_fn,
        float: collate_float_fn,
        int: collate_int_fn,
        str: collate_str_fn,
        bytes: collate_str_fn
    }

    return collate(batch, collate_fn_map=default_collate_fn_map)