"""
This module just prints the version of tensorflow and the number of GPUs available.
"""

import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))