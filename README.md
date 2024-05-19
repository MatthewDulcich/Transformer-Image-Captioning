Original by https://github.com/Dantekk/Image-Captioning/tree/main

# Image Captioning with CNN and Transformer

This repository contains the implementation of an Image Captioning application using Keras/Tensorflow. The application uses a Convolutional Neural Network (CNN) and a Transformer as encoder/decoder. 

## Architecture

The architecture consists of three models:

1. **A CNN**: EfficientNetB0 pre-trained on ImageNet is used to extract the image features.
2. **A TransformerEncoder**: The extracted image features are then passed to a Transformer encoder that generates a new representation of the inputs.
3. **A TransformerDecoder**: It takes the encoder output and the text data sequence as inputs and tries to learn to generate the caption.

## Dataset 

The model has been trained on the 2014 Train/Val COCO dataset. The dataset can be downloaded [here](https://cocodataset.org/#download). 

The original dataset has 82,783 train images and 40,504 validation images; for each image, there is a number of captions between 1 and 6. The dataset has been preprocessed to keep only images that have exactly 5 captions. After this filtering, the final dataset has 68,363 train images and 33,432 validation images.

The preprocessed dataset is serialized into two JSON files:

- `COCO_dataset/captions_mapping_train.json`
- `COCO_dataset/captions_mapping_valid.json`

Each element in the JSON files has the following structure:

```json
"COCO_dataset/train2014/COCO_train2014_000000318556.jpg": ["caption1", "caption2", "caption3", "caption4", "caption5"],
```

## API Key
Put your wandb api key in a file called `apikey.txt` or comment out the code

## Dependencies
I have used the following versions for code work:
* python==3.11.9
* tensorflow-macos==2.16.1
* tensorflow-metal==1.1.0
* numpy==1.19.1
* h5py==2.10.0
## Training
To train the model you need to follow the following steps :
1. you have to make sure that the training set images are in the folder `COCO_dataset/train2014/` and that validation set images are in `COCO_dataset/val2014/`.
2. you have to enter all the parameters necessary for the training in the `settings.py` file.
3. start the model training with `python3 training.py`
## Inference (a few different ways)
Run `inference.py`
Run `./log_inference.sh`
Run `./run_inference.sh`
### My settings
For my training session, I have get best results with this `settings.py` file :
```python
# Desired image dimensions
IMAGE_SIZE = (299, 299)
# Max vocabulary size
MAX_VOCAB_SIZE = 2000000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Number of self-attention heads
NUM_HEADS = 6
# Per-layer units in the feed-forward network
FF_DIM = 1024
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 14

# Reduce Dataset
# If you want reduce number of train/valid images dataset, set 'REDUCE_DATASET=True'
# and set number of train/valid images that you want.
#### COCO dataset
# Max number train dataset images : 68363
# Max number valid dataset images : 33432
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 68363]
NUM_TRAIN_IMG = 68363
# Number of valid images -> it must be a value between [1, 33432]
# N.B. -> IMPORTANT : the number of images of the test set is given by the difference between 33432 and NUM_VALID_IMG values.
# for instance, with NUM_VALID_IMG = 20000 -> valid set have 20000 images and test set have the last 13432 images.
NUM_VALID_IMG = 20000
# Data augumention on train set
TRAIN_SET_AUG = True
# Data augmention on valid set
VALID_SET_AUG = False
# If you want to calculate the performance on the test set.
TEST_SET = True

# Load train_data.json pathfile
train_data_json_path = "COCO_dataset/captions_mapping_train.json"
# Load valid_data.json pathfile
valid_data_json_path = "COCO_dataset/captions_mapping_valid.json"
# Load text_data.json pathfile
text_data_json_path  = "COCO_dataset/text_data.json"

# Save training files directory
SAVE_DIR = "save_train_dir/"
```
I have training model on full train set (68363 train images) and 20000 valid images but you can train the model on a smaller number of images by changing the NUM_TRAIN_IMG / NUM_VALID_IMG parameters to reduce the training time and hardware resources required.

### Data augmention
I applied data augmentation on the training set during the training to reduce the generalization error, with this transformations (this code is write in `dataset.py`) :
```python
trainAug = tf.keras.Sequential([
    	tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
    	tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)),
	tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10))
])
```
You can customize your data augmentation by changing this code or disable data augmentation setting `TRAIN_SET_AUG = False` in `setting.py`. 
### My results

These are the results on test set (13432 images):
```
loss: 11.8024 - acc: 0.5455
```

These are good results considering that for each image given as input to the model during training, **the error and the accuracy are averaged over 5 captions**. However, I spent little time doing model selection and you can improve the results by trying better settings. </br>
For example, you could :
1. change CNN architecture.
2. change SEQ_LENGTH, EMBED_DIM, NUM_HEADS, FF_DIM, BATCH_SIZE (etc...) parameters.
3. change data augmentation transformations/parameters.
4. change optimizer and learning rate scheduler.
5. etc...

**N.B.** I have saved my best training results files in the directory `save_train_dir/`.
## Inference
After training and saving the model, you can restore it in a new session to inference captions on new images. </br>
To generate a caption from a new image, you must :
1. insert the parameters in the file `settings_inference.py`
2. run `python3 inference.py --image={image_path_file}`

## Results example
Examples of image output taken from the test set.
| a large passenger jet flying through the sky             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/2.jpg)

| a man in a white shirt and black shorts playing tennis             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/10.jpg)  


| a person on a snowboard in the snow             |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/15.jpg)  

| a boy on a skateboard in the street            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/20.jpg)  

| a black bear is walking through the grass            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/4.jpg)  


| a train is on the tracks near a station            |  
:-------------------------:|
![](https://github.com/Dantekk/Image-Captioning/blob/main/examples_img/14.jpg)  
