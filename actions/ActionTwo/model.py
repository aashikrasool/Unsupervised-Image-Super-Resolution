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


os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
IMAGE_PATH = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR\\8.png"


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

load_image = preprocess_image(IMAGE_PATH)

super_image = model(load_image)
# super_image = tf.squeeze(super_image)
# super_image = np.asarray(super_image)
# super_image = tf.clip_by_value(super_image, 0, 255)
# super_image= Image.fromarray(tf.cast(super_image, tf.uint8).numpy())
# cv2.imwrite('action2_output/weighted3 super_image)

# print(super_image.shape)




# plot_image(tf.squeeze(super_image),'Super Resolution')
image=np.asarray(super_image)
image =tf.clip_by_value(image, 0, 255)

image=np.squeeze(image)

img = Image.fromarray(np.uint8(image))

folder_path = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output"
file_path = os.path.join(folder_path, "newo_image.png")
img.save(file_path)
image = cv2.imread(file_path)
new_size = (384, 384)
resized_image = cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)
cv2.imwrite(file_path, resized_image)




# img= np.array(img)
# cv2.imwrite(file_path,img)

# file_n="/content/drive/MyDrive/Dataset/SR"
#cv2.imwrite('/content/drive/MyDrive/Dataset/SR',image)
# # image = Image.fromarray(tf.cast(image, tf.uint8).numpy())