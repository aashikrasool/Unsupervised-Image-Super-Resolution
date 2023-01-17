# Importing basic libraries

#mortivated from https://www.kaggle.com/code/yasserh/esrgan-image-super-resolution
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2



def preprocess_image(image_path):
    '''Loads the image given make it ready for
      the model
      Args:
        image_path: Path to the image file
   '''
    image = tf.image.decode_image(tf.io.read_file(image_path))
    if image.shape[-1] == 4:
        image = image[..., :-1]
    size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)
