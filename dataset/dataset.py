import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import time
import concurrent.futures
import zipfile
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomDataset(Dataset):
    """
    Custom dataset class for handling custom data.
    This class extends the base Dataset class to create a custom dataset.
    """

    def __init__(self, folder1, folder2, target_size, batch_size):
        self.folder1 = folder1
        self.folder2 = folder2
        self.target_size = target_size
        self.batch_size = batch_size
        self.files1 = self.get_image_files_from_zip(folder1)
        self.files2 = self.get_image_files_from_zip(folder2)
        self.truth_images = []
        self.distorted_images = []
        self.load_images()

    def get_image_files_from_zip(self, zip_file_path):
        image_files = []
        with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
            for file_name in zip_file.namelist():
                if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                    image_files.append(file_name)
        return sorted(image_files)

    def process_image(self, file, folder):
        try:
            with zipfile.ZipFile(folder, 'r') as zip_file:
                with zip_file.open(file) as image_file:
                    image = Image.open(image_file)
                    image = image.resize(self.target_size)

                    # Ensure the image has 3 channels (RGB)
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    image = np.array(image)
                    image = np.transpose(image, (2, 0, 1))  # Adjust the channel dimension
                    return image
        except Exception as e:
            logger.error(f"Error processing image {file}: {e}")
            return None

    def load_images(self):
        start_time = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_truth_images = [
                    executor.submit(self.process_image, file, self.folder1) for file in self.files1
                ]
                future_distorted_images = [
                    executor.submit(self.process_image, file, self.folder2) for file in self.files2
                ]
                self.truth_images = [future.result() for future in concurrent.futures.as_completed(future_truth_images)]
                self.distorted_images = [future.result() for future in concurrent.futures.as_completed(future_distorted_images)]
        except Exception as e:
            logger.error("Exception occurred during image loading:", e)

        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Image loading time: %s seconds", execution_time)
        return self.truth_images, self.distorted_images

    def transform_distorted_images(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        transformed_distorted_images = []
        for image_ in self.distorted_images:
            if image_.shape[0] > 3:
                image_ = Image.fromarray(image_.transpose(1, 2, 0))
            else:
                image_ = Image.fromarray(image_.transpose(1, 2, 0), mode="RGB")

            transformed_distorted = transform(image_)
            transformed_distorted_images.append(transformed_distorted)

        return transformed_distorted_images

    def augment_images(self, images, augmentation_factor):
        augmented_images = []
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.6, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        for image in images:
            augmented_batch = []
            for _ in range(augmentation_factor):
                # Convert image to PIL format if it has more than 4 channels
                if image.shape[0] > 3:
                    image_pil = Image.fromarray(image.transpose(1, 2, 0))
                else:
                    image_pil = Image.fromarray(image.transpose(1, 2, 0), mode="RGB")
                augmented_image = transform(image_pil)
                augmented_batch.append(augmented_image)
            augmented_images.extend(augmented_batch)

        return augmented_images

    def augment_truth_images(self, augmentation_factor, num_lists):
        # Calculate the number of images per list
        num_images_per_list = len(self.truth_images) // num_lists
        start_time = time.time()
        augmented_truth_images = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_augmented_images = []
            for i in range(num_lists):
                batch_start = i * num_images_per_list
                batch_end = (i + 1) * num_images_per_list
                batch_images = self.truth_images[batch_start:batch_end]
                future = executor.submit(self.augment_images, batch_images, augmentation_factor)
                future_augmented_images.append(future)

            for future in concurrent.futures.as_completed(future_augmented_images):
                augmented_images = future.result()
                augmented_truth_images.extend(augmented_images)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Augmentation time: %s seconds", execution_time)
        return augmented_truth_images


    def __len__(self):
        return len(self.truth_images)

    def __getitem__(self, index):
        truth_image = self.truth_images[index]
        distorted_image = self.distorted_images[index]
        # Apply any necessary preprocessing or transformations here
        return truth_image, distorted_image
