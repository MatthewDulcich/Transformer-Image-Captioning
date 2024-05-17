"""
This module contains the paths to the saved tokenizer model, the configuration of the trained model, 
and the weights of the trained model. These paths are used during the inference process.

Attributes:
    tokenizer_path (str): The path to the saved tokenizer model. The tokenizer is used to convert 
                          the captions into a format that can be used by the model.
    get_model_config_path (str): The path to the configuration of the trained model. The configuration 
                                  includes information such as the architecture of the model and the 
                                  hyperparameters used during training.
    get_model_weights_path (str): The path to the weights of the trained model. The weights are the 
                                   learned parameters of the model.

Usage:
    These paths are used in the inference script to load the necessary components for generating 
    captions for new images. The tokenizer is used to process the generated captions, the configuration 
    is used to recreate the model architecture, and the weights are loaded into the recreated model 
    to allow it to generate captions.
"""

# Tokenizer model saved path
tokernizer_path = "save_train_dir/tokenizer.keras"
# Config model saved path
get_model_config_path = "save_train_dir/config_train.json"
# Weights model saved path
get_model_weights_path = "save_train_dir/big_model_weights_coco.weights.h5"
