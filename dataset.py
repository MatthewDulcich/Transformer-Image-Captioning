"""
dataset.py

This module contains functions for handling the dataset for the image captioning model.

It includes:
- Functions to split the dataset into training, validation, and test sets.
- Functions to reduce the dimension of the dataset.
- Functions to read and augment images.
- A function to create a TensorFlow dataset from the images and captions.

The module uses TensorFlow for data handling and image processing, and numpy for data manipulation.
"""

import re
import os
import math
import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from settings import *
import image_aug

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
AUTOTUNE = tf.data.AUTOTUNE

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    """
    Standardize the input string by converting it to lowercase and removing special characters.

    Args:
        input_string (str): The input string.

    Returns:
        str: The standardized string.
    """
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """
    Split the caption data into training and validation sets.

    Args:
        caption_data (dict): The caption data.
        train_size (float, optional): The proportion of the data to use for training. Defaults to 0.8.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.

    Returns:
        tuple: The training and validation data.
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data

def valid_test_split(captions_mapping_valid):
    """
    Split the validation data into validation and test sets.

    Args:
        captions_mapping_valid (dict): The validation data.

    Returns:
        tuple: The validation and test data.
    """
    valid_data={}
    test_data={}
    conta_valid = 0
    for id in captions_mapping_valid:
        if conta_valid<NUM_VALID_IMG:
            valid_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
        else:
            test_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
    return valid_data, test_data

def reduce_dataset_dim(captions_mapping_train, captions_mapping_valid):
    """
    Reduce the dimension of the training and validation data.

    Args:
        captions_mapping_train (dict): The training data.
        captions_mapping_valid (dict): The validation data.

    Returns:
        tuple: The reduced training and validation data.
    """
    train_data = {}
    conta_train = 0
    for id in captions_mapping_train:
        if conta_train<=NUM_TRAIN_IMG:
            train_data.update({id : captions_mapping_train[id]})
            conta_train+=1
        else:
            break

    valid_data = {}
    conta_valid = 0
    for id in captions_mapping_valid:
        if conta_valid<=NUM_VALID_IMG:
            valid_data.update({id : captions_mapping_valid[id]})
            conta_valid+=1
        else:
            break

    return train_data, valid_data

def read_image_inf(img_path):
    """
    Read an image from a file and preprocess it.

    Args:
        img_path (str): The path to the image file.

    Returns:
        tf.Tensor: The preprocessed image.
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img

def read_image(data_aug):
    """
    Create a function to read and optionally augment an image.

    Args:
        data_aug (bool): Whether to augment the image.

    Returns:
        function: The function to read and optionally augment an image.
    """
    def decode_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)

        if data_aug:
            img = augment(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def augment(img):
        img = tf.expand_dims(img, axis=0)
        img = img_transf(img)
        img = tf.squeeze(img, axis=0)
        return img

    return decode_image

img_transf = tf.keras.Sequential([
            	tf.keras.layers.RandomContrast(factor=(0.05, 0.15)),
                #image_aug.RandomBrightness(brightness_delta=(-0.15, 0.15)),
                #image_aug.PowerLawTransform(gamma=(0.8,1.2)),
                #image_aug.RandomSaturation(sat=(0, 2)),
                #image_aug.RandomHue(hue=(0, 0.15)),
                #tf.keras.layers.RandomFlip("horizontal"),
	    	    tf.keras.layers.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
		        tf.keras.layers.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
		        tf.keras.layers.RandomRotation(factor=(-0.10, 0.10))])

def make_dataset(images, captions, data_aug, tokenizer):
    """
    Create a TensorFlow dataset from the images and captions.

    Args:
        images (list): The images.
        captions (list): The captions.
        data_aug (bool): Whether to augment the images.
        tokenizer (tf.keras.layers.TextVectorization): The tokenizer.

    Returns:
        tf.data.Dataset: The TensorFlow dataset.
    """
    read_image_xx = read_image(data_aug)
    img_dataset = tf.data.Dataset.from_tensor_slices(images)

    img_dataset = (img_dataset
                   .map(read_image_xx, num_parallel_calls=AUTOTUNE))

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(tokenizer, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset
