"""
custom_schedule.py

This module contains a custom learning rate schedule for the image captioning model.

The custom schedule is a class that inherits from TensorFlow's LearningRateSchedule. It implements a learning rate schedule that varies depending on the number of training steps.

The module uses TensorFlow for the implementation of the learning rate schedule.
"""

import tensorflow as tf

#@tf.keras.utils.register_keras_serializable()
class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   """
    Custom learning rate schedule.

    Args:
        d_model (int): The base learning rate.
        warmup_steps (int, optional): The number of warmup steps. Defaults to 4000.
   """
   def __init__(self, d_model, warmup_steps=4000):
      """
        Initialize the custom learning rate schedule.

        Args:
            d_model (int): The base learning rate.
            warmup_steps (int, optional): The number of warmup steps. Defaults to 4000.
      """
      super(custom_schedule, self).__init__()
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps

   def __call__(self, step):
      """
        Calculate the learning rate for a given step.

        Args:
            step (int): The current training step.

        Returns:
            float: The learning rate for the given step.
      """
      arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
      arg2 = tf.cast(step, tf.float32) * (self.warmup_steps ** -1.5)
      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

   def get_config(self):
      config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps
        }
      return config
