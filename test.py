import tensorflow as tf

print(tf.__version__)
print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))