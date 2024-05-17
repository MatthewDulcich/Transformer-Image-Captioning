"""
image_aug.py

This module contains classes for various image augmentation operations.

It includes:
- RandomBrightness: Adjusts the brightness of images by a random factor.
- PowerLawTransform: Applies a power law (gamma) transform to images with a random exponent.
- RandomSaturation: Adjusts the saturation of images by a random factor.
- RandomHue: Adjusts the hue of images by a random factor.

The module uses TensorFlow for the image augmentation operations and numpy for random number generation.
"""

import tensorflow as tf
import numpy as np

class RandomBrightness(tf.keras.layers.Layer):
    """
    Adjusts the brightness of images by a random factor.

    Args:
        brightness_delta (tuple): The range of brightness factors.
    """
    def __init__(self, brightness_delta, **kwargs):
        """
        Initialize the RandomBrightness layer.

        Args:
            brightness_delta (tuple): The range of brightness factors.
            **kwargs: Additional keyword arguments.
        """
        super(RandomBrightness, self).__init__(**kwargs)
        self.brightness_delta = brightness_delta

    def call(self, images, training=None):
        """
        Adjust the brightness of the images.

        Args:
            images (tf.Tensor): The input images.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The brightness-adjusted images.
        """
        #if not training:
        #    return images

        brightness = np.random.uniform(self.brightness_delta[0], self.brightness_delta[1])

        images = tf.image.adjust_brightness(images, brightness)
        return images

class PowerLawTransform(tf.keras.layers.Layer):
    """
    Applies a power law (gamma) transform to images with a random exponent.

    Args:
        gamma (tuple): The range of gamma values.
    """
    def __init__(self, gamma, **kwargs):
        """
        Initialize the RandomSaturation layer.

        Args:
            sat (tuple): The range of saturation factors.
            **kwargs: Additional keyword arguments.
        """
        super(PowerLawTransform, self).__init__(**kwargs)
        self.gamma = gamma

    def call(self, images, training=None):
        """
        Adjust the saturation of the images.

        Args:
            images (tf.Tensor): The input images.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The saturation-adjusted images.
        """
        #if not training:
        #    return images

        gamma_value = np.random.uniform(self.gamma[0], self.gamma[1])

        images = tf.image.adjust_gamma(images, gamma_value)
        return images

class RandomSaturation(tf.keras.layers.Layer):
    """
    Adjusts the saturation of images by a random factor.

    Args:
        sat (tuple): The range of saturation factors.
    """
    def __init__(self, sat, **kwargs):
        super(RandomSaturation, self).__init__(**kwargs)
        self.sat = sat

    def call(self, images, training=None):
        #if not training:
        #    return images

        #sat_value = np.random.uniform(self.sat[0], self.sat[1])

        images = tf.image.random_saturation(images, self.sat[0], self.sat[1])
        return images

class RandomHue(tf.keras.layers.Layer):
    """
    Adjusts the hue of images by a random factor.

    Args:
        hue (tuple): The range of hue factors.
    """
    def __init__(self, hue, **kwargs):
        """
        Initialize the RandomHue layer.

        Args:
            hue (tuple): The range of hue factors.
            **kwargs: Additional keyword arguments.
        """        
        super(RandomHue, self).__init__(**kwargs)
        self.hue = hue

    def call(self, images, training=None):
        """
        Adjust the hue of the images.

        Args:
            images (tf.Tensor): The input images.
            training (bool, optional): Whether the layer is in training mode. Defaults to None.

        Returns:
            tf.Tensor: The hue-adjusted images.
        """
        #if not training:
        #    return images

        #hue_value = np.random.uniform(self.hue[0], self.hue[1])

        images = tf.image.random_hue(images, self.hue[0], self.hue[1])
        return images
