# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Test collate function."""

import collections
from types import MappingProxyType

import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import DataLoader, Dataset, default_collate, default_convert
from tests.mark_utils import arg_mark


class ImmutableMapping(collections.abc.Mapping):
    """
    An immutable mapping.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class UnsupportedMutableMapping(collections.abc.MutableMapping):
    """
    A mutable mapping that is not supported by the default collate function.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    # deliberately not implement copy method
    def __copy__(self):
        raise TypeError("not support copy operation")


class MutableMappingWithExtraProperty(collections.abc.MutableMapping):
    """A mutable mapping with some extra properties."""

    def __init__(self, data, extra_property=None):
        self._data = dict(data)
        self.extra_property = extra_property

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.extra_property == other.extra_property and self._data == other._data


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_default_convert_class_type():
    """
    Feature: Test default_convert function.
    Description: Test default_convert function with ImmutableMapping and UnsupportedMutableMapping.
    Expectation: The result is as expected.
    """
    dataloader1 = default_convert(ImmutableMapping({0: "a", 1: "b"}))
    print(dataloader1)
    dataloader2 = default_convert(UnsupportedMutableMapping({0: "x", 1: "y"}))
    print(dataloader2)
    assert dataloader2 == {0: "x", 1: "y"}


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_default_convert_common_type():
    """
    Feature: Test default_convert function.
    Description: Test default_convert function with Tensor, number, Type SaUO (byte, string, unicode, object).
    Expectation: The result is as expected.
    """
    # Tensor
    assert default_convert(ms.Tensor(1)) == ms.Tensor(1)

    # number
    assert default_convert(1) == 1
    assert default_convert(np.array([1])) == ms.Tensor([1])
    assert default_convert(np.array(1)) == ms.Tensor(1)

    # Type SaUO (byte, string, unicode, object)
    assert default_convert("abc") == "abc"
    assert default_convert(np.array("abc")) == np.array("abc")

    assert default_convert(b"abc") == b"abc"
    assert default_convert(np.array(b"abc")) == np.array(b"abc")

    data = [1, 2.5, "hello", [1, 2, 3], {"a": 1, "b": 2}]
    array_object = np.array(data, dtype=object)
    assert (default_convert(array_object) == array_object).all()

    # dict
    def compare_dict(d1, d2):
        assert d1.keys() == d2.keys()
        for k in d1.keys():
            if isinstance(d1[k], ms.Tensor) and isinstance(d2[k], ms.Tensor):
                assert (d1[k] == d2[k]).all()
            else:
                assert d1[k] == d2[k]

    data = {"a": 1, "b": 2, "c": 3}
    compare_dict(default_convert(data), data)

    data = {"a": np.array(1), "b": np.ones((2, 3)), "c": np.array([4, 5, 6])}
    expected = {
        "a": ms.Tensor(data["a"]),
        "b": ms.Tensor(data["b"]),
        "c": ms.Tensor(data["c"]),
    }
    compare_dict(default_convert(data), expected)

    # list
    def compare_seq(l1, l2):
        assert len(l1) == len(l2)
        for v1, v2 in zip(l1, l2):
            if isinstance(v1, dict):
                v1, v2 = v1["d"], v2["d"]
            if isinstance(v1, ms.Tensor) and isinstance(v2, ms.Tensor):
                assert (v1 == v2).all()
            else:
                assert v1 == v2

    data = [1, np.array(2), [3], {"d": np.array([4, 5])}]
    expected = [1, ms.Tensor(2), [3], {"d": ms.Tensor([4, 5])}]
    compare_seq(default_convert(data), expected)

    # tuple
    data = (1, np.array(2), [3], {"d": np.array([4, 5])})
    expected = (1, ms.Tensor(2), [3], {"d": ms.Tensor([4, 5])})
    compare_seq(default_convert(data), expected)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_default_collate():
    """
    Feature: Test default_collate function.
    Description: Test default_collate function with same type and different type.
    Expectation: The result is as expected.
    """
    # same type
    inputs = [
        ms.Tensor(np.array(1, dtype=np.uint8)),
        ms.Tensor(np.array(0, dtype=np.uint8)),
    ]
    assert (default_collate(inputs) == ms.Tensor([1, 0])).all()

    # different type, not support by ops
    inputs = [
        ms.Tensor(np.array(1, dtype=np.uint8)),
        ms.Tensor(np.array(2, dtype=np.float32)),
    ]
    assert (default_collate(inputs) == ms.Tensor([1, 2])).all()


class MyDataset(Dataset):
    """
    A map style dataset that returns as many samples as requested.
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = list(range(num_samples))

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_collate_fn():
    """
    Feature: Test collate_fn function.
    Description: Test collate_fn function with MyDataset.
    Expectation: The result is as expected.
    """

    def my_collate(data):
        return [{"ori plus one": d + 1} for d in data]

    dataset = MyDataset(10)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate)
    for data in dataloader:
        print(data)


class TestDefaultConvert:
    """Test default_convert function."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            ms.Tensor(True, dtype=ms.bool),
            ms.Tensor([0], dtype=ms.uint8),
            ms.Tensor([[-1], [1]], dtype=ms.int32),
            ms.Tensor([[[3.14], [-3.14]], [[3.14], [-3.14]]], dtype=ms.float32),
            ms.Tensor([1 + 2j], dtype=ms.complex64),
        ),
    )
    def test_convert_tensor(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with Tensor.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data.asnumpy(), result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.bool_(True),
            np.uint8(0),
            np.int32(-1),
            np.float32(3.14),
            np.complex64(1 + 2j),
        ),
    )
    def test_convert_numpy_scalar(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy primitive type that should be converted.
        Expectation: The result is a Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data, result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize("data", (np.str_("abc"), np.bytes_(b"0xffff")))
    def test_convert_unsupported_numpy_scalar(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy primitive type that should not be converted.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert result == data

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.array([True], dtype=np.bool_),
            np.array([0], dtype=np.uint8),
            np.array([[-1], [1]], dtype=np.int32),
            np.array([[[3.14], [-3.14]], [[3.14], [-3.14]]], dtype=np.float32),
            np.array([1 + 2j], dtype=np.complex64),
        ),
    )
    def test_convert_numpy_array(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy array that should be converted.
        Expectation: The result is a Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, ms.Tensor)
        np.testing.assert_equal(data, result.asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            np.array([b"abc"], dtype=np.bytes_),
            np.array(["abc"], dtype=np.str_),
            np.array([{"data": "abc"}], dtype=np.object_),
        ),
    )
    def test_convert_unsupported_numpy_array(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with numpy array that should not be converted.
        Expectation: The result is unchanged.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        np.testing.assert_equal(data, result)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            MutableMappingWithExtraProperty(
                {"uint8": np.uint8(0), "int32": np.int32(-1), "float32": np.float32(3.14)}, extra_property="converted"
            ),
            MutableMappingWithExtraProperty({"str": "a", "bytes": b"b"}, extra_property="unchanged"),
        ),
    )
    def test_convert_mutable_mapping_with_extra_property(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mutable mapping with extra property that should be converted.
        Expectation: The result is a mapping with values converted to Tensor and extra property is preserved.
        """
        result = default_convert(data)
        assert isinstance(result, MutableMappingWithExtraProperty)
        assert data == result

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    @pytest.mark.parametrize(
        "data",
        (
            {"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)},
            collections.OrderedDict({"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
            collections.defaultdict(np.int32, {"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
            MappingProxyType({"1": np.int32(1), "2": np.int32(2), "3": np.int32(3)}),
        ),
    )
    def test_convert_mapping(self, data):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mutable mapping that should be converted.
        Expectation: The result is a mapping with values converted to Tensor.
        """
        result = default_convert(data)
        assert isinstance(result, type(data))
        assert data.keys() == result.keys()
        for key, value in data.items():
            assert isinstance(result[key], ms.Tensor)
            np.testing.assert_equal(value, result[key].asnumpy())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_convert_mapping_not_support_copy(self):
        """
        Feature: Test default_convert function.
        Description: Test default_convert function with mutable mapping that should be converted.
        Expectation: The result is a mapping with values converted to Tensor.
        """

        def unimplemented_copy(self):
            raise TypeError

        MutableMappingWithExtraProperty.__copy__ = unimplemented_copy
        data = MutableMappingWithExtraProperty({"uint8": np.uint8(0)}, extra_property="converted")
        result = default_convert(data)
        assert isinstance(result, dict)
        assert data.keys() == result.keys()
        for key, value in data.items():
            assert isinstance(result[key], ms.Tensor)
            np.testing.assert_equal(value, result[key].asnumpy())
