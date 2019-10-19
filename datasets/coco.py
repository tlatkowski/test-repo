import io
import json
import os

import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow.python.keras import preprocessing
import tensorflow_datasets as tfds


logging.set_verbosity(logging.INFO)

CAPTIONS_DIR = '../annotations/captions_val2017.json'
IMAGES_DIR = '../val2017'


def load_json_captions(path: str):
    caption_file = io.open(path)
    caption_json = json.load(caption_file)
    return caption_json


def load_data(captions_path=CAPTIONS_DIR, images_path=IMAGES_DIR):
    captions_json = load_json_captions(captions_path)
    annotations = captions_json["annotations"]
    
    images_paths = []
    captions_texts = []
    for annotation in annotations:
        image_id = annotation["image_id"]
        caption_txt = annotation["caption"]
        image_fn = os.path.join(images_path, "{:0>12}.jpg".format(image_id))
        
        images_paths.append(image_fn)
        captions_texts.append(caption_txt)
    
    images_paths = np.array(images_paths)
    captions_texts = np.array(captions_texts)
    
    return images_paths, captions_texts


def load_vactorized_data(captions_path=CAPTIONS_DIR, images_path=IMAGES_DIR):
    captions_json = load_json_captions(captions_path)
    
    annotations = captions_json["annotations"]
    tokenizer = preprocessing.text.Tokenizer(num_words=10)
    
    images_paths = []
    captions_texts = []
    logging.info('model saved to ')
    for annotation in annotations:
        image_id = annotation["image_id"]
        caption_txt = annotation["caption"]
        
        # tokenizer.fit_on_texts(caption_txt)
        # vectorized_caption = tokenizer.texts_to_sequences(caption_txt)
        image_path = os.path.join(images_path, "{:0>12}.jpg".format(image_id))
        images_paths.append(image_path)
        # captions_texts.append(np.array(vectorized_caption))
        captions_texts.append(caption_txt)
    
    images_paths = np.array(images_paths)
    captions_texts = np.array(captions_texts)
    
    return images_paths, captions_texts


IMG_WIDTH = 64
IMG_HEIGHT = 64


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    img *= 255.0
    img = (img - 127.5) / 127.5
    # resize the image to the desired size.
    return tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))





def coco_dataset_iterator(captions_path=CAPTIONS_DIR, images_path=IMAGES_DIR):
    images_paths, captions_texts = load_vactorized_data(captions_path, images_path)
    
    vocab = set()
    tokenizer = tfds.features.text.Tokenizer()
    max_len = 0
    for text in captions_texts:
        tokens = tokenizer.tokenize(text)
        vocab.update(tokens)
        if len(tokens) > max_len:
            max_len = len(tokens)
    encoder = tfds.features.text.TokenTextEncoder(vocab)

    encoded = []
    for text in captions_texts:
        e = encoder.encode(text)
        e_padded = np.pad(np.array(e), pad_width=(0, max_len - len(e)))
        encoded.append(e_padded)
    
    def parse_text_and_load_img(text, image):
        # encoded_text = encoder.encode(text)
        img = tf.io.read_file(image)
        img = decode_img(img)
        return text, img
    print(len(images_paths)/64)
    dataset = tf.data.Dataset.from_tensor_slices((encoded, images_paths))
    dataset = dataset.map(parse_text_and_load_img)
    # iterator = dataset.make_one_shot_iterator()
    dataset = dataset.shuffle(10)
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(10)
    # iterator = dataset.make_one_shot_iterator()

    # i = next(iter(dataset))
    # print(i)
    return dataset, max_len, len(vocab)