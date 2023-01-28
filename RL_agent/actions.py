# building esrgan method is motivated from https://www.kaggle.com/code/yasserh/esrgan-image-super-resolution
# imported pre-trained model from https://tfhub.dev/captain-pool/esrgan-tf2/1


import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import cv2

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)
out_folder="this is for saving output esrgan"

# this pre-process is just for  action 2 esrgan
def preprocess_image(img):
    '''Loads the image given make it ready for
      the model
      Args:
        image_path: Path to the image file
   '''
    image = tf.image.decode_image(tf.io.read_file(img))
    if image.shape[-1] == 4:
        image = image[..., :-1]
    size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
    image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)


def downsample(img_file, scale=0.3, plot=False):
    img = cv2.imread(img_file, 1)
    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return img, img_small


def build_hr(model_path, scale, modelname1, img_small):
    model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
    print("Reading model file {}".format(model_path))
    model_pretrained.readModel(model_path)
    model1 = model_pretrained.setModel(modelname1, scale)

    img_upscaled1 = model1.upsample(img_small)

    return img_upscaled1


def action1(img_small, config):
    img_upscaled1 = build_hr(img_small, config['edsr_model'], "edsr", 4)
    img_upscaled2 = build_hr(img_small, config['espcn_model'], "espcn", 4)
    comb1 = cv2.addWeighted(img_upscaled1, 0.6, img_upscaled2, 0.4, 0.0)
    return comb1


def action3(img_small, config):
    img_upscaled1 = build_hr(img_small, config['lapsrn_model'], "lapsrn", 4)
    img_upscaled2 = build_hr(img_small, config['espcn_model'], "espcn", 4)
    comb2 = cv2.addWeighted(img_upscaled1, 0.6, img_upscaled2, 0.4, 0.0)
    return comb2


def action2(pre_processed_img, SAVED_MODEL_PATH, out_folder):
    model_new = hub.load(SAVED_MODEL_PATH)
    super_image = model_new(pre_processed_img)
    image = np.asarray(super_image)
    image = tf.cast(image, tf.uint8)

    image = np.squeeze(image)

    img = Image.fromarray(np.uint8(image))

    folder_path = out_folder
    file_path = os.path.join(folder_path, "esrgan_out.png")
    img.save(file_path)
    image = cv2.imread(file_path)
    new_size = (384, 384)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(file_path, resized_image)
    msg = print("esrgan Image saved")
    return msg,resized_image


