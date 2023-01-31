# building esrgan method is motivated from https://www.kaggle.com/code/yasserh/esrgan-image-super-resolution
# imported pre-trained model from https://tfhub.dev/captain-pool/esrgan-tf2/1


import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageOps
import tensorflow as tf
import tensorflow_hub as hub
import cv2

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = 'True'
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
model = hub.load(SAVED_MODEL_PATH)

# this pre-process is just for  action 2 esrgan
def preprocess_image(img):
    '''Loads the image given make it ready for
      the model
      Args:
        image_path: Path to the image file
   '''
    image = tf.image.decode_image(tf.io.read_file(img))
    if image.shape[-1] == 4:#for RGBA images
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

def action2(lr_img,SAVED_MODEL_PATH,time_stamp):
    out_folder="G:\\Gachon Masters\\pycharm\\reinforcement"
    pre_processed_img=preprocess_image(lr_img)
    model_new =hub.load(SAVED_MODEL_PATH)
    super_image = model_new(pre_processed_img)
    image = np.asarray(super_image)
    image = tf.cast(image, tf.uint8)

    image = np.squeeze(image)

    img = Image.fromarray(np.uint8(image))

    folder_path = out_folder
    file_path = os.path.join(folder_path, f"{out_folder}\\{time_stamp}esrgan_out.png")
    img.save(file_path)
    #msg=print("esrgan Image saved")
    return img

def action3(img_small, config,):
    img_upscaled1 = build_hr(img_small, config['lapsrn_model'], "lapsrn", 4)
    img_upscaled2 = build_hr(img_small, config['espcn_model'], "espcn", 4)
    comb2 = cv2.addWeighted(img_upscaled1, 0.6, img_upscaled2, 0.4, 0.0)
    return comb2

def action4(img,time_stamp):
    out_folder = "G:\\Gachon Masters\\pycharm\\reinforcement"
    if time_stamp==0:
        img = Image.open(img)
        img = ImageOps.fit(img, (img.width * 4, img.height * 4), method=Image.Resampling.NEAREST)
        curr_bri = ImageEnhance.Brightness(img)
        new_bri = 1.08
        brightened = curr_bri.enhance(new_bri)


        brightened.save(f"{out_folder}\\{time_stamp}upsampled.jpg")

        return brightened
    else:
        img = Image.open(img)
        curr_bri = ImageEnhance.Brightness(img)
        new_bri = 1.08
        brightened = curr_bri.enhance(new_bri)
        brightened.save(f"{out_folder}\\{time_stamp}out_act4.jpg")
        return brightened
    return brightened

def action5(img,time_stamp):
    out_folder = "G:\\Gachon Masters\\pycharm\\reinforcement"
    if time_stamp == 0:
        img = Image.open(img)
        img = ImageOps.fit(img, (img.width * 4, img.height * 4), method=Image.Resampling.NEAREST)
        curr_bri = ImageEnhance.Brightness(img)
        new_bri = 0.92
        brightened = curr_bri.enhance(new_bri)

        brightened.save(f"{out_folder}\\{time_stamp}upsampled.jpg")

        return brightened
    else:
        img = Image.open(img)
        curr_bri = ImageEnhance.Brightness(img)
        new_bri = 0.92
        brightened = curr_bri.enhance(new_bri)
        brightened.save(f"{out_folder}\\{time_stamp}out_act4.jpg")
        return brightened
    return brightened

def action6(img,time_stamp):
    out_folder = "G:\\Gachon Masters\\pycharm\\reinforcement"
    if time_stamp == 0:
        img = Image.open(img)
        img = ImageOps.fit(img, (img.width * 4, img.height * 4), method=Image.Resampling.NEAREST)
        curr_sharp =  ImageEnhance.Sharpness(img)
        sharpened_img = curr_sharp.enhance(1.08)
        sharpened_img.save(f"{out_folder}\\{time_stamp}upsampled.jpg")

        return sharpened_img
    else:
        img = Image.open(img)
        curr_sharp = ImageEnhance.Sharpness(img)
        sharpened_img = curr_sharp.enhance(1.08)
        sharpened_img.save(f"{out_folder}\\{time_stamp}act_6.jpg")
        return sharpened_img
    return sharpened_img







