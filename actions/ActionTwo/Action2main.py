#mortivated from https://www.kaggle.com/code/yasserh/esrgan-image-super-resolution
#impoted pre trained model from https://tfhub.dev/captain-pool/esrgan-tf2/1
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from model import preprocess_image


os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
IMAGE_PATH = "C:\\Users\\aashi\\PycharmProjects\\Unsupervised-Image-Super-Resolution\\LR\\0.png"

out_folder="G:\\Gachon Masters\\pycharm\\reinforcement"
def action2(pre_processed_img,SAVED_MODEL_PATH ,out_folder):

    model_new =hub.load(SAVED_MODEL_PATH)
    super_image = model_new(pre_processed_img)
    image = np.asarray(super_image)
    image = tf.cast(image, tf.uint8)

    image = np.squeeze(image)

    img = Image.fromarray(np.uint8(image))

    folder_path = out_folder
    file_path = os.path.join(folder_path, "esrgan_out.png")
    img.save(file_path)
    msg=print("esrgan Image saved")
    return msg
img=preprocess_image(IMAGE_PATH )
actions= action2(img,SAVED_MODEL_PATH,out_folder)

