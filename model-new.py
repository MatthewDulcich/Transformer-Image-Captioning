"""
This module contains the implementation of an Image Captioning Model. The model is composed of several parts:

1. A CNN model for image feature extraction.
2. Two encoder layers for transforming the extracted features.
3. A decoder for generating the caption.

The CNN model is used to extract features from the input images. The extracted features are then passed through two
encoder layers to transform them into a suitable format for the decoder. The decoder takes these transformed features
and generates a caption for the input image.

The model also includes a loss tracker and an accuracy tracker for monitoring the training process. The loss tracker
keeps track of the loss during training, and the accuracy tracker keeps track of the accuracy.

The module includes the following classes:

- `PositionalEmbedding`: This class represents a Positional Embedding layer.
- `TransformerEncoderBlock`: This class represents a Transformer Encoder Block.
- `TransformerEncoder`: This class represents a Transformer Encoder.
- `TransformerDecoderBlock`: This class represents a Transformer Decoder Block.
- `ImageCaptioningModel`: This class represents an Image Captioning Model.

Each class has its own attributes and methods, which are documented in their respective docstrings.

This module can be run as a standalone script to train the Image Captioning Model on a dataset of images and captions.
The training process includes the following steps:

1. Load the dataset.
2. Preprocess the images and captions.
3. Train the model on the preprocessed data.
4. Evaluate the model on a test set.
5. Save the trained model.

The module requires the following packages: tensorflow, keras, and numpy.
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
    This function creates a CNN model using EfficientNetB0 as the base model.
    The base model is frozen and reshaped to match the required output.

    Returns:
        cnn_model (keras.Model): The constructed CNN model.
    """
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    # Freeze feature extractor layers
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, 1280))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model

class TransformerEncoderBlock(layers.Layer):
    """
    This class represents a Transformer Encoder Block. It contains a multi-head attention layer,
    a dense layer, and a layer normalization.
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

    def call(self, inputs, training, mask=None):
        """
        This method is called when the layer is invoked.

        Args:
            inputs (Tensor): The input tensor.
            training (bool): Whether the model is in training mode.
            mask (Tensor, optional): The mask tensor.

        Returns:
            proj_input (Tensor): The output tensor.
        """
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=None
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input


class PositionalEmbedding(layers.Layer):
    """
    This class represents a Positional Embedding layer. It contains two types of embeddings:
    token embeddings and position embeddings. The token embeddings convert the input tokens
    into vectors of specified dimension (embed_dim). The position embeddings convert the 
    position of each token into vectors of specified dimension (embed_dim). The final output 
    is the sum of these two embeddings.

    Attributes:
        token_embeddings (layers.Embedding): The token embedding layer.
        position_embeddings (layers.Embedding): The position embedding layer.
        sequence_length (int): The length of the sequence.
        vocab_size (int): The size of the vocabulary.
        embed_dim (int): The dimension of the embedding.
    """
    
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        """
        The constructor for PositionalEmbedding class.

        Args:
            sequence_length (int): The length of the sequence.
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the embedding.
        """
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
        This method is called when the layer is invoked.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor, which is the sum of token and position embeddings.
        """
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        """
        This method computes a mask tensor. This mask tensor is the same shape as the input tensor
        and indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.

        Args:
            inputs (Tensor): The input tensor.
            mask (Tensor, optional): An optional mask tensor.

        Returns:
            Tensor: A mask tensor of the same shape as the input.
        """
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    """
    This class represents a Transformer Decoder Block. It contains two multi-head attention layers,
    a dense layer, and three layer normalizations. It also includes a PositionalEmbedding layer and
    a final dense layer for output. Dropout is applied after the embedding layer and the third layer normalization.

    Attributes:
        embed_dim (int): The dimension of the embedding.
        ff_dim (int): The dimension of the feed forward network.
        num_heads (int): The number of attention heads.
        vocab_size (int): The size of the vocabulary.
        attention_1 (layers.MultiHeadAttention): The first multi-head attention layer.
        attention_2 (layers.MultiHeadAttention): The second multi-head attention layer.
        dense_proj (keras.Sequential): The dense layer.
        layernorm_1 (layers.LayerNormalization): The first layer normalization.
        layernorm_2 (layers.LayerNormalization): The second layer normalization.
        layernorm_3 (layers.LayerNormalization): The third layer normalization.
        embedding (PositionalEmbedding): The positional embedding layer.
        out (layers.Dense): The output layer.
        dropout_1 (layers.Dropout): The first dropout layer.
        dropout_2 (layers.Dropout): The second dropout layer.
        supports_masking (bool): Whether this layer supports masking.
    """
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size, **kwargs):
        """
        The constructor for TransformerDecoderBlock class.

        Args:
            embed_dim (int): The dimension of the embedding.
            ff_dim (int): The dimension of the feed forward network.
            num_heads (int): The number of attention heads.
            vocab_size (int): The size of the vocabulary.
        """
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
        This method is called when the layer is invoked.

        Args:
            inputs (Tensor): The input tensor.
            encoder_outputs (Tensor): The output tensor from the encoder.
            training (bool): Whether the model is in training mode.
            mask (Tensor, optional): The mask tensor.

        Returns:
            preds (Tensor): The output tensor.
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
        This method generates a causal attention mask, which is used to ensure that the attention
        mechanism does not look ahead during training.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: A causal attention mask.
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
    This class represents an Image Captioning Model. It contains a CNN model for image feature extraction,
    two encoder layers for transforming the extracted features, and a decoder for generating the caption.
    It also includes a loss tracker and an accuracy tracker for monitoring the training process.

    Attributes:
        cnn_model (keras.Model): The CNN model for image feature extraction.
        encoder (keras.Model): The first encoder layer.
        encoder2 (keras.Model): The second encoder layer.
        decoder (keras.Model): The decoder for generating the caption.
        loss_tracker (keras.metrics.Mean): The loss tracker.
        acc_tracker (keras.metrics.Mean): The accuracy tracker.
        num_captions_per_image (int): The number of captions per image.
    """
    def __init__(
        self, cnn_model, encoder, encoder2, decoder, num_captions_per_image=5,
    ):
        """
        The constructor for ImageCaptioningModel class.

        Args:
            cnn_model (keras.Model): The CNN model for image feature extraction.
            encoder (keras.Model): The first encoder layer.
            encoder2 (keras.Model): The second encoder layer.
            decoder (keras.Model): The decoder for generating the caption.
            num_captions_per_image (int, optional): The number of captions per image. Defaults to 5.
        """
        super().__init__()0
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.encoder2 = encoder2
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image


    def call(self, inputs):
        """
        This method is called when the model is invoked.

        Args:
            inputs (list): The list of inputs, which includes the image and the training flag.

        Returns:
            Tensor: The output tensor.
        """
        x = self.cnn_model(inputs[0])
        x = self.encoder(x, training=False)  # Pass training as a keyword argument
        # x = self.encoder(x, training=inputs[1]) # Pass training as a keyword argument
        x = self.decoder(inputs[1], x, training=False, mask=None)  # Pass training as a keyword argument
        # x = self.decoder(inputs[2], x, training=inputs[1], mask=None) # Pass training as a keyword argument

        return x

    def calculate_loss(self, y_true, y_pred, mask):
        """
        This method calculates the loss.

        Args:
            y_true (Tensor): The ground truth tensor.
            y_pred (Tensor): The predicted tensor.
            mask (Tensor): The mask tensor.

        Returns:
            Tensor: The loss tensor.
        """
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        """
        This method calculates the accuracy.

        Args:
            y_true (Tensor): The ground truth tensor.
            y_pred (Tensor): The predicted tensor.
            mask (Tensor): The mask tensor.

        Returns:
            Tensor: The accuracy tensor.
        """
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def train_step(self, batch_data):
        """
        This method is called for each batch of data during training.

        Args:
            batch_data (tuple): The batch data, which includes the image and the sequence.

        Returns:
            dict: A dictionary that contains the loss and accuracy.
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

    def test_step(self, batch_data):
        """
        This method is called for each batch of data during testing.

        Args:
            batch_data (tuple): The batch data, which includes the image and the sequence.

        Returns:
            dict: A dictionary that contains the loss, accuracy, predicted captions, and true captions.
        """
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        predicted_captions=[]
        true_captions=[]


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
            predictions = self.decoder(batch_seq_inp, encoder_out2, training=False)
            predicted_tokens=tf.argmax(predictions,axis=-1)
        

            # 6. Calculate loss and accuracy
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            # 7. Update the batch loss and batch accuracy
            batch_loss += caption_loss
            batch_acc += caption_acc
            for pred,true in zip(predicted_tokens,batch_seq_inp):
                predicted_text = self.tokenizer.decode(pred.numpy())
                true_text = self.tokenizer.decode(true.numpy())
                predicted_captions.append(predicted_text)
                true_captions.append(true_text)

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result(),'predicted_captions':predicted_captions,'true_captions':true_captions}

    @property
    def metrics(self):
        """
        This method returns the list of metrics.

        Returns:
            list: The list of metrics.
        """
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
