"""
model.py

This module contains the implementation of the image captioning model.

It includes:
- A function to load a pre-trained CNN model.
- A Transformer Encoder Block class for the transformer architecture.
- A Transformer Decoder Block class for the transformer architecture.
- An ImageCaptioningModel class that combines the CNN model and the Transformer blocks.

The CNN model is used for image feature extraction. The Transformer Encoder and Decoder Blocks are used for sequence modeling. The ImageCaptioningModel class combines these components into a complete model.

The module uses TensorFlow for model creation and training.

The settings for the model are imported from the settings module.
"""

import tensorflow as tf
import os
import certifi
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.applications import efficientnet
from settings import *

def get_cnn_model():
    """
    Function to load a pre-trained CNN model.

    Returns:
        keras.Model: Pre-trained CNN model.
    """
    # base_model = efficientnet.EfficientNetB0(
    #     input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    # # Freeze feature extractor layers
    # base_model.trainable = False
    # base_model_out = base_model.output
    # base_model_out = layers.Reshape((-1, 1280))(base_model_out)
    # cnn_model = keras.models.Model(base_model.input, base_model_out)

    # load reshaped_finetuned_model.keras model
    cnn_model = keras.models.load_model('reshaped_finetuned_model.keras')
    return cnn_model

class TransformerEncoderBlock(layers.Layer):
    """
    Transformer Encoder Block class.

    Args:
        embed_dim (int): Dimension of the embedding.
        dense_dim (int): Dimension of the dense layer.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        """
        Call function for the Transformer Encoder Block.

        Args:
            inputs (Tensor): Input tensor.
            training (bool): Whether the model is in training mode.
            mask (Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tensor: Output tensor.
        """
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        proj_input = self.layernorm_2(inputs + attention_output)
        return proj_input


class PositionalEmbedding(layers.Layer):
    """
    Positional Embedding class.

    Args:
        sequence_length (int): Length of the sequence.
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embedding.
    """
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        """
        Call function for the Positional Embedding.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        """
        Computes mask for the Positional Embedding.

        Args:
            inputs (Tensor): Input tensor.
            mask (Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tensor: Mask tensor.
        """
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """
    Transformer Decoder Block class.

    Args:
        embed_dim (int): Dimension of the embedding.
        ff_dim (int): Dimension of the feed forward layer.
        num_heads (int): Number of attention heads.
        vocab_size (int): Size of the vocabulary.
    """
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=self.vocab_size
        )
        self.out = layers.Dense(self.vocab_size)
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True


    def call(self, inputs, encoder_outputs, training, mask=None):
        """
        Call function for the Transformer Decoder Block.

        Args:
            inputs (Tensor): Input tensor.
            encoder_outputs (Tensor): Output tensor from the encoder.
            training (bool): Whether the model is in training mode.
            mask (Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tensor: Output tensor.
        """
        inputs = self.embedding(inputs)
        inputs = self.dropout_1(inputs, training=training)

        combined_mask = None
        padding_mask  = None

        if mask is not None:
            causal_mask = self.get_causal_attention_mask(inputs)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)


        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=combined_mask # None
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask#None
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.dropout_2(proj_out, training=training)

        preds = self.out(proj_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        """
        Computes the causal attention mask.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Causal attention mask.
        """
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class ImageCaptioningModel(keras.Model):
    """
    Image Captioning Model class.

    Args:
        cnn_model (keras.Model): CNN model for image feature extraction.
        encoder (keras.Layer): Encoder layer.
        encoder2 (keras.Layer): Second encoder layer.
        decoder (keras.Layer): Decoder layer.
        num_captions_per_image (int, optional): Number of captions per image. Defaults to 5.
    """
    def __init__(
        self, cnn_model, encoder, encoder2, decoder, num_captions_per_image=5 # removed tokenizer
    ):
        """
        Initialize the Image Captioning Model.

        Args:
            cnn_model (keras.Model): CNN model for image feature extraction.
            encoder (keras.Layer): Encoder layer.
            encoder2 (keras.Layer): Second encoder layer.
            decoder (keras.Layer): Decoder layer.
            num_captions_per_image (int, optional): Number of captions per image. Defaults to 5.
        """
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.encoder2 = encoder2
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.accumulation_steps = 5
        #self.tokenizer = tokenizer

    @tf.function
    def call(self, inputs):
        """
        Call function for the Image Captioning Model.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, training=False)  # Pass training as a keyword argument
        # x = self.encoder(x, training=inputs[1]) # Pass training as a keyword argument
        x = self.decoder(inputs[1], x, training=False, mask=None)  # Pass training as a keyword argument
        # x = self.decoder(inputs[2], x, training=inputs[1], mask=None) # Pass training as a keyword argument
        return x

    # @tf.function
    # def calculate_loss(self, y_true, y_pred, mask):
    #     loss = self.loss(y_true, y_pred)
    #     mask = tf.cast(mask, dtype=loss.dtype)
    #     loss *= mask
    #     return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    # @tf.function
    def calculate_loss(self, y_true, y_pred, mask):
        """
        Calculate the loss for the model.

        Args:
            y_true (Tensor): True labels.
            y_pred (Tensor): Predicted labels.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Loss tensor.
        """
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        # Reshape the output of tf.argmax into a 1D tensor
        y_pred_argmax = tf.reshape(tf.argmax(y_pred, axis=2), [-1])

        # Compute penalty for repeated words
        # _, _, count = tf.unique_with_counts(y_pred_argmax)
        # penalty = tf.reduce_sum(tf.cast(count > 1, dtype=loss.dtype))

        return tf.reduce_sum(loss) / tf.reduce_sum(mask) #+ (0.3 * penalty)

    @tf.function
    def calculate_accuracy(self, y_true, y_pred, mask):
        """
        Calculate the accuracy for the model.

        Args:
            y_true (Tensor): True labels.
            y_pred (Tensor): Predicted labels.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Accuracy tensor.
        """
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    @tf.function
    def train_step(self, batch_data):
        """
        Perform one training step.

        Args:
            batch_data (tuple): Tuple of batch image and sequence data.

        Returns:
            dict: Dictionary with loss and accuracy.
        """
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                # 3. Pass image embeddings to encoder
                encoder_out = self.encoder(img_embed, training=True)
                encoder_out2 = self.encoder2(encoder_out, training=True)  # New encoder block

                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]

                # 4. Compute the mask for the input sequence
                mask = tf.math.not_equal(batch_seq_inp, 0)

                # 5. Pass the encoder outputs, sequence inputs along with
                # mask to the decoder
                # batch_seq_pred1 = self.decoder(batch_seq_inp, encoder_out, training=True, mask=mask)
                batch_seq_pred2 = self.decoder(batch_seq_inp, encoder_out2, training=True, mask=mask)  # New encoder block


                # 6. Calculate loss and accuracy
                caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred2, mask)
                caption_acc = self.calculate_accuracy(
                    batch_seq_true, batch_seq_pred2, mask
                )

                # 7. Update the batch loss and batch accuracy
                batch_loss += caption_loss
                batch_acc += caption_acc

            # 8. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.encoder2.trainable_variables + self.decoder.trainable_variables
            )

            # 9. Get the gradients
            grads = tape.gradient(caption_loss, train_vars)

            # 10. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


    # # @tf.function
    # def train_step(self, batch_data):
    #     batch_img, batch_seq = batch_data
    #     batch_loss = 0
    #     batch_acc = 0

    #     # Initialize accumulated gradients
    #     accumulated_grads = [tf.zeros_like(w) for w in self.trainable_weights]

    #     # 1. Get image embeddings
    #     img_embed = self.cnn_model(batch_img)

    #     # 2. Pass each of the five captions one by one to the decoder
    #     # along with the encoder outputs and compute the loss as well as accuracy
    #     # for each caption.
    #     for i in range(self.num_captions_per_image):
    #         with tf.GradientTape() as tape:
    #              # 3. Pass image embeddings to encoder
    #             encoder_out = self.encoder(img_embed, training=True)
    #             encoder_out2 = self.encoder2(encoder_out, training=True)  # New encoder block

    #             batch_seq_inp = batch_seq[:, i, :-1]
    #             batch_seq_true = batch_seq[:, i, 1:]

    #             # 4. Compute the mask for the input sequence
    #             mask = tf.math.not_equal(batch_seq_inp, 0)

    #             # 5. Pass the encoder outputs, sequence inputs along with
    #             # mask to the decoder
    #             # batch_seq_pred1 = self.decoder(batch_seq_inp, encoder_out, training=True, mask=mask)
    #             batch_seq_pred2 = self.decoder(batch_seq_inp, encoder_out2, training=True, mask=mask)  # New encoder block

    #             # 6. Calculate loss and accuracy
    #             caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred2, mask)
    #             caption_acc = self.calculate_accuracy(
    #                 batch_seq_true, batch_seq_pred2, mask
    #             )

    #             # 7. Update the batch loss and batch accuracy
    #             batch_loss += caption_loss
    #             batch_acc += caption_acc

    #             # 8. Get the list of all the trainable weights
    #             train_vars = (
    #                 self.encoder.trainable_variables + self.encoder2.trainable_variables + self.decoder.trainable_variables
    #             )

    #              # 9. Get the gradients
    #             grads = tape.gradient(caption_loss, train_vars)

    #             # Check if gradients are None or NaN
    #             if any(g is None or tf.reduce_any(tf.math.is_nan(g)) for g in grads):
    #                 raise ValueError("Gradients are None or NaN.")

    #             # Accumulate gradients instead of applying them directly
    #             accumulated_grads = [acc_g + g for acc_g, g in zip(accumulated_grads, grads)]

    #         # Apply the accumulated gradients and reset them to zero every accumulation_steps
    #         if (i + 1) % self.accumulation_steps == 0:
    #             if any(g is None or tf.reduce_any(tf.math.is_nan(g)) for g in accumulated_grads):
    #                 raise ValueError("Accumulated gradients are None or NaN.")
    #             self.optimizer.apply_gradients(zip(accumulated_grads, train_vars))
    #             accumulated_grads = [tf.zeros_like(w) for w in self.trainable_weights]

    #     # Apply any remaining accumulated gradients
    #     if accumulated_grads and any(tf.reduce_sum(g) for g in accumulated_grads):
    #         self.optimizer.apply_gradients(zip(accumulated_grads, train_vars))

    #     loss = batch_loss
    #     acc = batch_acc / float(self.num_captions_per_image)
    #     # Update the trackers
    #     self.loss_tracker.update_state(loss)
    #     self.acc_tracker.update_state(acc)

    #     # Return the results
    #     return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}


    @tf.function
    def test_step(self, batch_data):
        """
        Perform one testing step.

        Args:
            batch_data (tuple): Tuple of batch image and sequence data.

        Returns:
            dict: Dictionary with loss and accuracy.
        """
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            # 3. Pass image embeddings to encoder
            encoder_out = self.encoder(img_embed, training=False)
            encoder_out2 = self.encoder2(encoder_out, training=False)  # New encoder block

            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]

            # 4. Compute the mask for the input sequence
            mask = tf.math.not_equal(batch_seq_inp, 0)

            # 5. Pass the encoder outputs, sequence inputs along with
            # mask to the decoder
            # batch_seq_pred = self.decoder(
            #     batch_seq_inp, encoder_out, training=False, mask=mask
            # )
            batch_seq_pred = self.decoder(batch_seq_inp, encoder_out2, training=False, mask=mask)  # New encoder block

            # 6. Calculate loss and accuracy
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            # 7. Update the batch loss and batch accuracy
            batch_loss += caption_loss
            batch_acc += caption_acc

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    # def test_step(self, batch_data):
    #         batch_img, batch_seq = batch_data
    #         batch_loss = 0
    #         batch_acc = 0
    #         predicted_captions=[]
    #         true_captions=[]


    #         # 1. Get image embeddings
    #         img_embed = self.cnn_model(batch_img)

    #         # 2. Pass each of the five captions one by one to the decoder
    #         # along with the encoder outputs and compute the loss as well as accuracy
    #         # for each caption.
    #         for i in range(self.num_captions_per_image):
    #             # 3. Pass image embeddings to encoder
    #             encoder_out = self.encoder(img_embed, training=False)
    #             encoder_out2 = self.encoder2(encoder_out, training=False)  # New encoder block

    #             batch_seq_inp = batch_seq[:, i, :-1]
    #             batch_seq_true = batch_seq[:, i, 1:]

    #             # 4. Compute the mask for the input sequence
    #             mask = tf.math.not_equal(batch_seq_inp, 0)

    #             # 5. Pass the encoder outputs, sequence inputs along with
    #             # mask to the decoder
    #             # batch_seq_pred = self.decoder(
    #             #     batch_seq_inp, encoder_out, training=False, mask=mask
    #             # )
    #             batch_seq_pred = self.decoder(batch_seq_inp, encoder_out2, training=False, mask=mask)  # New encoder block
    #             predictions = self.decoder(batch_seq_inp, encoder_out2, training=False)
    #             predicted_tokens=tf.argmax(predictions,axis=-1)

    #             # 6. Calculate loss and accuracy
    #             caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
    #             caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

    #             # 7. Update the batch loss and batch accuracy
    #             batch_loss += caption_loss
    #             batch_acc += caption_acc
    #             # for pred,true in zip(predicted_tokens,batch_seq_inp):
    #             #     # predicted_text = self.tokenizer.decode(pred.numpy())
    #             #     # true_text = self.tokenizer.decode(true.numpy())
    #             #     predicted_text = decode_numerical_tokens(pred.numpy(), self.tokenizer)
    #             #     true_text = decode_numerical_tokens(true.numpy(), self.tokenizer)
    #             #     predicted_captions.append(predicted_text)
    #             #     true_captions.append(true_text)

    #             predicted_tokens_list = tf.unstack(predicted_tokens)
    #             batch_seq_inp_list = tf.unstack(batch_seq_inp)

    #             for pred, true in zip(predicted_tokens_list, batch_seq_inp_list):
    #                 predicted_text = tf.py_function(decode_numerical_tokens, [pred, self.tokenizer], tf.string)
    #                 true_text = tf.py_function(decode_numerical_tokens, [true, self.tokenizer], tf.string)
    #                 predicted_captions.append(predicted_text)
    #                 true_captions.append(true_text)


    #         loss = batch_loss
    #         acc = batch_acc / float(self.num_captions_per_image)

    #         self.loss_tracker.update_state(loss)
    #         self.acc_tracker.update_state(acc)
    #         return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result(),'predicted_captions':predicted_captions,'true_captions':true_captions}




    @property
    def metrics(self):
        """
        List of the metrics to be tracked.

        Returns:
            list: List of metrics.
        """
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]

def decode_numerical_tokens(numerical_tokens, vectorization_layer):
    """
    Decode numerical tokens back to words.

    Args:
        numerical_tokens (list): List of numerical tokens to be decoded.
        vectorization_layer (Layer): The vectorization layer used for encoding.

    Returns:
        str: Decoded string.
    """
    vocabulary = vectorization_layer.get_vocabulary()
    words = [vocabulary[token] for token in numerical_tokens]
    return ' '.join(words)

def process_token(pred, true):
    """
    Process predicted and true tokens to convert them into text.

    Args:
        pred (list): List of predicted numerical tokens.
        true (list): List of true numerical tokens.

    Returns:
        tuple: Tuple of predicted and true text.
    """
    predicted_text = tf.py_function(decode_numerical_tokens, [pred, self.tokenizer], tf.string)
    true_text = tf.py_function(decode_numerical_tokens, [true, self.tokenizer], tf.string)
    return predicted_text, true_text
