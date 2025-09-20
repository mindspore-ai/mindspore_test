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
# ============================================================================


from mindspore.parallel.mpmd.pipeline_parallel._utils import BatchDimSpec
import unittest


class TestSlotsBehavior(unittest.TestCase):

    def test_cannot_add_new_attributes(self):
        """
        Description:Test that new attributes cannot be dynamically added
        Expectation:Run success
        """
        spec = BatchDimSpec(0)

        with self.assertRaises(AttributeError):
            spec.new_attribute = "test_value"

        with self.assertRaises(AttributeError):
            spec.another_attribute = 123

    def test_can_access_defined_attributes(self):
        """
        Description:Test that defined attributes can be accessed
        Expectation:Run success
        """
        spec = BatchDimSpec(0)

        self.assertEqual(spec.batch_dim, 0)

        spec.batch_dim = 5
        self.assertEqual(spec.batch_dim, 5)

    def test_multiple_instances_independence(self):
        """Test attribute independence between multiple instances"""
        spec1 = BatchDimSpec(0)
        spec2 = BatchDimSpec(1)

        spec1.batch_dim = 10
        spec2.batch_dim = 20

        self.assertEqual(spec1.batch_dim, 10)
        self.assertEqual(spec2.batch_dim, 20)

        with self.assertRaises(AttributeError):
            spec1.new_attr = "fail"

        with self.assertRaises(AttributeError):
            spec2.new_attr = "also_fail"

        # Both instances' original attributes should remain unchanged
        self.assertEqual(spec1.batch_dim, 10)
        self.assertEqual(spec2.batch_dim, 20)

    def test_edge_cases(self):

        # Test specific edge cases for BatchDimSpec
        spec = BatchDimSpec(0)

        # Try setting existing attribute to None
        spec.batch_dim = None
        self.assertIsNone(spec.batch_dim)

        # Reset to normal value
        spec.batch_dim = 5
        self.assertEqual(spec.batch_dim, 5)


class TestBatchDimSpecSlotsIntegration(unittest.TestCase):
    """Integration test: Test slots performance of BatchDimSpec in actual usage"""

    def test_slots_in_data_structures(self):
        """Test behavior of slots objects in data structures"""
        specs = [BatchDimSpec(i) for i in range(5)]

        self.assertEqual(len(specs), 5)
        self.assertEqual(specs[2].batch_dim, 2)


class TestBatchDimSpecStaticMethods(unittest.TestCase):

    def test_from_tuple_valid_input(self):
        """
        Description: Test from_tuple with valid tuple input
        Expectation: Should return a tuple of BatchDimSpec objects with correct batch_dim values
        """
        # Test with normal tuple
        input_tuple = (0, 1, 2)
        result = BatchDimSpec.from_tuple(input_tuple)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        # Check each element is BatchDimSpec with correct batch_dim
        for i, spec in enumerate(result):
            self.assertIsInstance(spec, BatchDimSpec)
            self.assertEqual(spec.batch_dim, i)

    def test_from_tuple_empty_input(self):
        """
        Description: Test from_tuple with empty tuple
        Expectation: Should return empty tuple
        """
        result = BatchDimSpec.from_tuple(())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 0)

    def test_from_tuple_single_element(self):
        """
        Description: Test from_tuple with single element tuple
        Expectation: Should return tuple with single BatchDimSpec
        """
        result = BatchDimSpec.from_tuple((5,))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], BatchDimSpec)
        self.assertEqual(result[0].batch_dim, 5)

    def test_from_tuple_negative_values(self):
        """
        Description: Test from_tuple with negative batch dimensions
        Expectation: Should handle negative values correctly
        """
        result = BatchDimSpec.from_tuple((-1, -2))
        self.assertEqual(result[0].batch_dim, -1)
        self.assertEqual(result[1].batch_dim, -2)

    def test_from_tuple_invalid_input_type(self):
        """
        Description: Test from_tuple with invalid input type
        Expectation: Should raise TypeError
        """
        with self.assertRaises(TypeError):
            BatchDimSpec.from_tuple([0, 1, 2])  # List instead of tuple

        with self.assertRaises(TypeError):
            BatchDimSpec.from_tuple("string")  # String instead of tuple

        with self.assertRaises(TypeError):
            BatchDimSpec.from_tuple(123)  # Integer instead of tuple

    def test_from_dict_valid_input(self):
        """
        Description: Test from_dict with valid dictionary input
        Expectation: Should return a dictionary with BatchDimSpec objects as values
        """
        input_dict = {"input1": 0, "input2": 1, "input3": 2}
        result = BatchDimSpec.from_dict(input_dict)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)

        # Check keys are preserved and values are BatchDimSpec objects
        for key, spec in result.items():
            self.assertIn(key, input_dict)
            self.assertIsInstance(spec, BatchDimSpec)
            self.assertEqual(spec.batch_dim, input_dict[key])

    def test_from_dict_empty_input(self):
        """
        Description: Test from_dict with empty dictionary
        Expectation: Should return empty dictionary
        """
        result = BatchDimSpec.from_dict({})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_from_dict_single_element(self):
        """
        Description: Test from_dict with single key-value pair
        Expectation: Should return dictionary with single BatchDimSpec value
        """
        result = BatchDimSpec.from_dict({"tensor": 3})
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result["tensor"], BatchDimSpec)
        self.assertEqual(result["tensor"].batch_dim, 3)

    def test_from_dict_negative_values(self):
        """
        Description: Test from_dict with negative batch dimensions
        Expectation: Should handle negative values correctly
        """
        result = BatchDimSpec.from_dict({"a": -1, "b": -2})
        self.assertEqual(result["a"].batch_dim, -1)
        self.assertEqual(result["b"].batch_dim, -2)

    def test_from_dict_invalid_input_type(self):
        """
        Description: Test from_dict with invalid input type
        Expectation: Should raise TypeError
        """
        with self.assertRaises(TypeError):
            BatchDimSpec.from_dict([("a", 0), ("b", 1)])  # List instead of dict

        with self.assertRaises(TypeError):
            BatchDimSpec.from_dict("string")  # String instead of dict

        with self.assertRaises(TypeError):
            BatchDimSpec.from_dict(123)  # Integer instead of dict

    def test_from_dict_complex_keys(self):
        """
        Description: Test from_dict with various key types
        Expectation: Should handle different key types correctly
        """
        # Test with integer keys
        result = BatchDimSpec.from_dict({0: 1, 1: 2})
        self.assertEqual(result[0].batch_dim, 1)
        self.assertEqual(result[1].batch_dim, 2)

        # Test with tuple keys
        result = BatchDimSpec.from_dict({(0, 1): 0, (1, 2): 1})
        self.assertEqual(result[(0, 1)].batch_dim, 0)
        self.assertEqual(result[(1, 2)].batch_dim, 1)

    def test_round_trip_behavior(self):
        """
        Description: Test that objects created by static methods behave like regular BatchDimSpec
        Expectation: Should have same behavior and constraints
        """
        # Test from_tuple
        tuple_result = BatchDimSpec.from_tuple((0, 1))[0]
        regular_spec = BatchDimSpec(0)

        # Both should have same attributes and behavior
        self.assertEqual(tuple_result.batch_dim, regular_spec.batch_dim)

        # Both should have slots constraint
        with self.assertRaises(AttributeError):
            tuple_result.new_attr = "test"

        with self.assertRaises(AttributeError):
            regular_spec.new_attr = "test"

        # Test from_dict
        dict_result = BatchDimSpec.from_dict({"test": 5})["test"]
        self.assertEqual(dict_result.batch_dim, 5)

        with self.assertRaises(AttributeError):
            dict_result.new_attr = "test"


class TestBatchDimSpecIntegration(unittest.TestCase):
    """Integration tests for BatchDimSpec static methods"""

    def test_combined_usage(self):
        """
        Description: Test using both static methods together in a pipeline scenario
        Expectation: Should work correctly in integrated scenarios
        """
        # Simulate a pipeline configuration
        input_batch_dims = BatchDimSpec.from_dict({
            "encoder_input": 0,
            "decoder_input": 0
        })

        hidden_batch_dims = BatchDimSpec.from_tuple((0, 1))

        # Verify the structure
        self.assertIsInstance(input_batch_dims, dict)
        self.assertIsInstance(hidden_batch_dims, tuple)

        # Verify specific values
        self.assertEqual(input_batch_dims["encoder_input"].batch_dim, 0)
        self.assertEqual(hidden_batch_dims[1].batch_dim, 1)

    def test_repr_and_str_consistency(self):
        """
        Description: Test that objects from static methods have consistent string representations
        Expectation: repr() and str() should work the same as regular instances
        """
        tuple_spec = BatchDimSpec.from_tuple((3,))[0]
        dict_spec = BatchDimSpec.from_dict({"test": 3})["test"]
        regular_spec = BatchDimSpec(3)

        # All should have same string representations
        self.assertEqual(repr(tuple_spec), repr(regular_spec))
        self.assertEqual(str(tuple_spec), str(regular_spec))
        self.assertEqual(repr(dict_spec), repr(regular_spec))
        self.assertEqual(str(dict_spec), str(regular_spec))


if __name__ == '__main__':
    unittest.main(verbosity=2)
