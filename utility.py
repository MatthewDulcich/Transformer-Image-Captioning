"""
training.py

This module contains the training pipeline for the image captioning model.

It includes:
- Importing necessary modules and settings.
- Defining a callback for logging training metrics with Weights & Biases (wandb).
- Reading the API key for wandb from a file and logging in.
- The main training script will be further down in the file (not shown in the excerpt).

The module uses TensorFlow for model training, and Weights & Biases for logging training metrics.

The settings for the model and the dataset are imported from the settings and dataset modules respectively. The model components are imported from the model module.
"""

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from custom_schedule import custom_schedule
from tensorflow import keras
from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from dataset import read_image_inf
import numpy as np
import json
import re
from settings import *

def save_tokenizer(tokenizer, path_save):
    """
    Save the tokenizer to a file.

    Args:
        tokenizer (TextVectorization): The tokenizer to be saved.
        path_save (str): The path where the tokenizer should be saved.

    Returns:
        None
    """
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = tokenizer(input)
    model = tf.keras.Model(input, output)
    model.save(path_save + "tokenizer.h5")

def get_inference_model(model_config_path):
    """
    Get the inference model.

    Args:
        model_config_path (str): The path to the model configuration file.

    Returns:
        ImageCaptioningModel: The inference model.
    """
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    encoder2 = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, encoder2=encoder2, decoder=decoder
    )

    ##### It's necessary for init model -> without it, weights subclass model fails
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, decoder_input]) 
    # caption_model([cnn_input, training, decoder_input])

    #####

    return caption_model


def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH):
    """
    Generate a caption for an image.

    Args:
        image_path (str): The path to the image.
        caption_model (ImageCaptioningModel): The model to use for caption generation.
        tokenizer (TextVectorization): The tokenizer to use for tokenization.
        SEQ_LENGTH (int): The maximum length of a sequence.

    Returns:
        str: The generated caption.
    """
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # Read the image from the disk
    img = read_image_inf(image_path)

    # Pass the image to the CNN
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "sos "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        #tokenized_caption = tokenizer.predict([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "eos":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("sos ", "")
