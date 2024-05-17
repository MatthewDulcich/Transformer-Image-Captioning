"""
finetuning.py

This module contains the code for fine-tuning an EfficientNetB0 model on the COCO dataset.

It includes:
- Functions to preprocess image paths and draw bounding boxes.
- Code to load and preprocess the COCO dataset.
- Code to define, compile, and train the model.
- Code to save the trained model and plot the training history.
- Code to reshape the model's output layer and save the reshaped model.

The module uses TensorFlow for model definition and training, pandas for data manipulation, 
matplotlib for plotting, cv2 for image processing, and wandb for logging training metrics.
"""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
import wandb

BATCH_SIZE = 32

# Read the API key from the file
with open('apikey_thilak.txt', 'r') as file:
    api_key = file.read().strip()

run = wandb.init(project='Fine Tuning Efficient Net', entity='thilak-cm212')

# Login to wandb
wandb.login(key=api_key)

# Define functions to preprocess image paths and draw bounding boxes
def add_zeros_and_extension(image_ids):
    """
    Add leading zeros and the .jpg extension to each image ID.

    Args:
        image_ids (list): The image IDs.

    Returns:
        list: The image IDs with leading zeros and the .jpg extension.
    """
    return [f'{str(image_id).zfill(12)}.jpg' for image_id in image_ids]

def draw_bounding_boxes(image_ids, anns_df, images_dir, num_images=5):
    """
    Draw bounding boxes on a random sample of images.

    Args:
        image_ids (list): The image IDs.
        anns_df (pd.DataFrame): The DataFrame containing the annotations.
        images_dir (str): The directory containing the images.
        num_images (int, optional): The number of images to draw bounding boxes on. Defaults to 5.
    """
    random_image_ids = random.sample(list(image_ids), num_images)
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(20, 6))
    for i, image_id in enumerate(random_image_ids):
        image_path = os.path.join(images_dir, image_id)
        image = cv2.imread(image_path)
        bbox = anns_df.loc[anns_df['file_name'] == image_id, 'bbox'].values[0]
        bbox = [int(coord) for coord in bbox]
        x, y, w, h = bbox
        image_with_bbox = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        axes[i].imshow(cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1}")
    plt.tight_layout()
    plt.show()

# Define paths
anns_file_path = 'coco/annotations/instances_val2017.json'
images_dir = 'coco/images/val2017'

# Load annotations
with open(anns_file_path, 'r') as f:
    anns = json.loads(f.read())

# Convert annotations to DataFrame
anns_df = pd.DataFrame(anns['annotations'])
anns_df.drop_duplicates(subset=['image_id'], inplace=True)
anns_df['file_name'] = add_zeros_and_extension(anns_df['image_id'])

# Display bounding boxes
draw_bounding_boxes(anns_df['file_name'], anns_df, images_dir, num_images=5)

# Define data paths
image_dir = 'coco/images/val2017/'
anns_df['file_name'] = image_dir + anns_df['file_name'].astype(str)

# Define image processing functions
def preprocess_image(image_path, label):
    """
    Preprocess an image for the model.

    Args:
        image_path (str): The path to the image.
        label (int): The label of the image.

    Returns:
        tuple: The preprocessed image and its label.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

# Perform train-test split
train_df, val_df = train_test_split(anns_df, test_size=0.1, random_state=42)

# Convert labels to one-hot encoded format
num_classes = 91
train_labels = tf.keras.utils.to_categorical(train_df['category_id'].values, num_classes=num_classes)
val_labels = tf.keras.utils.to_categorical(val_df['category_id'].values, num_classes=num_classes)

# Create TensorFlow Dataset objects with labels
train_ds = tf.data.Dataset.from_tensor_slices((train_df['file_name'].values, train_labels))
train_ds = train_ds.map(preprocess_image).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((val_df['file_name'].values, val_labels))
val_ds = val_ds.map(preprocess_image).batch(BATCH_SIZE)

# Define model architecture
base_model = EfficientNetB0(input_shape=(299, 299, 3), include_top=False, weights="imagenet")
base_model.trainable = False
for layer in base_model.layers[-10:]:
    layer.trainable = True
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(num_classes, activation='softmax')(x)
cnn_model = models.Model(base_model.input, output)

# Compile model
cnn_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss=Huber(),
                  metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Define WandB callback
wandb_callback = LambdaCallback(
    on_batch_end=lambda batch, logs: wandb.log({
        'batch_train_loss': logs['loss'],
        'batch_train_accuracy': logs['accuracy']
    }),
    on_epoch_end=lambda epoch, logs: wandb.log({
        'epoch_train_loss': logs['loss'],
        'epoch_valid_loss': logs.get('val_loss', None),
        'epoch_valid_accuracy': logs.get('val_accuracy', None)
    })
)

# Train model with early stopping and WandB callback
history = cnn_model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[early_stopping, wandb_callback])

# Save model
cnn_model.save('fine_tuned_efficientnet.keras')

# Plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Load and reshape model
model = tf.keras.models.load_model('fine_tuned_efficientnet.keras')
reshaped_output = tf.keras.layers.Reshape((-1, 1280))(model.get_layer('top_activation').output)
new_model = tf.keras.models.Model(inputs=model.input, outputs=reshaped_output)
new_model.save('reshaped_finetuned_model.keras')
print('Model reshaped and saved as reshaped_finetuned_ model.keras')
