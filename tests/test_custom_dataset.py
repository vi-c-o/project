import unittest
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import time
import concurrent.futures
import zipfile
import logging

# Append the project directory to sys.path
project_dir = os.path.abspath(os.path.dirname('../'))
sys.path.append(project_dir)

from dataset.dataset import CustomDataset

class CustomDatasetTestCase(unittest.TestCase):
   """
    Test case for the CustomDataset class.
    This class defines unit tests for the CustomDataset class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up test data and parameters
        cls.folder1 = "../dataset/images.zip"
        cls.folder2 = "../dataset/distorted.zip"
        cls.target_size = (128, 128)
        cls.batch_size = 16

        # Read the files from the zip folders
        cls.files1 = cls.get_image_files_from_zip(cls.folder1)
        cls.files2 = cls.get_image_files_from_zip(cls.folder2)

    @classmethod
    def get_image_files_from_zip(cls, zip_file_path):
        image_files = []
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                    image_files.append(file_name)
        return sorted(image_files)

    def setUp(self):
        # Initialize the dataset
        self.dataset = CustomDataset(self.folder1, self.folder2, self.target_size, self.batch_size)

    def test_dataset_loading(self):

        self.assertGreater(len(self.dataset), 0)
        self.assertEqual(len(self.dataset.truth_images), len(self.dataset.distorted_images))

    def test_dataset_transforms(self):

        transformed_distorted_images = self.dataset.transform_distorted_images()
        self.assertEqual(len(transformed_distorted_images), len(self.dataset.distorted_images))

    def test_dataset_augmentation(self):

        augmentation_factor = 4
        num_lists = 2
        augmented_truth_images = self.dataset.augment_truth_images(augmentation_factor, num_lists)
        expected_length = len(self.dataset.truth_images) * augmentation_factor -4
        self.assertEqual(len(augmented_truth_images), expected_length)

    def test_len(self):

        self.assertEqual(len(self.dataset), len(self.dataset.truth_images))

    def test_getitem(self):

        index = 0
        truth_image, distorted_image = self.dataset[index]
        self.assertIsNotNone(truth_image)
        self.assertIsNotNone(distorted_image)

    def test_process_image(self):

        file = self.files1[0]  
        folder = self.folder1
        image = self.dataset.process_image(file, folder)
        self.assertIsNotNone(image)

    def test_load_images(self):

        self.dataset.load_images()
        self.assertGreater(len(self.dataset.truth_images), 0)
        self.assertGreater(len(self.dataset.distorted_images), 0)

if __name__ == '__main__':
    unittest.main()

