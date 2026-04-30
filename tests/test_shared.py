import unittest

import SCTools._shared as shared


class SparseDatasetCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.public_sparse_dataset = shared._public_sparse_dataset
        self.legacy_sparse_dataset = shared._legacy_sparse_dataset

    def tearDown(self):
        shared._public_sparse_dataset = self.public_sparse_dataset
        shared._legacy_sparse_dataset = self.legacy_sparse_dataset

    def test_sparse_dataset_prefers_public_factory(self):
        def public_factory(group):
            return ("public", group)

        def legacy_factory(group):
            return ("legacy", group)

        shared._public_sparse_dataset = public_factory
        shared._legacy_sparse_dataset = legacy_factory

        self.assertEqual(shared._as_sparse_dataset("X"), ("public", "X"))

    def test_sparse_dataset_falls_back_to_legacy_factory(self):
        def legacy_factory(group):
            return ("legacy", group)

        shared._public_sparse_dataset = None
        shared._legacy_sparse_dataset = legacy_factory

        self.assertEqual(shared._as_sparse_dataset("X"), ("legacy", "X"))

    def test_sparse_dataset_errors_when_no_factory_exists(self):
        shared._public_sparse_dataset = None
        shared._legacy_sparse_dataset = None

        with self.assertRaises(ImportError):
            shared._as_sparse_dataset("X")


if __name__ == "__main__":
    unittest.main()
