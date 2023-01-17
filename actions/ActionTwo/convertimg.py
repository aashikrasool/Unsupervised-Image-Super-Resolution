import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
IMAGE_PATH = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR\\8.png"

image = tf.image.decode_image(tf.io.read_file(IMAGE_PATH))
if image.shape[-1] == 4:
    image = image[..., :-1]
size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
image = tf.cast(image, tf.float32)
image = tf.expand_dims(image, 0)

super_image = model(image)
super_image = tf.clip_by_value(super_image, 0, 255)
super_image = tf.squeeze(super_image)
super_image = super_image.numpy()
super_image = super_image.transpose()

new_size = (384, 384)
# image = image.numpy()

# Resize the image
new_size = (384, 384)
resized_image = cv2.resize(super_image, new_size, interpolation = cv2.INTER_LINEAR)
resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
# Save the image
folder_path = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output"
file_path = os.path.join(folder_path, "n_image.png")
cv2.imwrite(file_path, resized_image)