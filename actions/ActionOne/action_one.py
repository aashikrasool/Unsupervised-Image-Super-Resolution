import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# from utilities.manage_config import *
# from utilities.gaussian import comb_img
from utilities.metrics_calculator import psnr_calc
from utilities.helpers import downsample
from argparse import ArgumentParser
from actions.action_one import *

# Main pipeline file
#
# # fetch configuration
# parser = ArgumentParser()
# parser.add_argument('--config', type=str, default='./configs/config.yaml', help="testing configuration file")
# args = parser.parse_args()
# config = get_config(args.config)
#
#
# # Removed platform-dependent code
# project_root = os.getcwd()
# dir_dataset = config['data_dir']


dir_dataset="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR"
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]


original_image, img_small = downsample(files_img[8], scale=0.4, plot=False)

dir_pretrained_models="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\pretrained_models"
# dir_pretrained_models = config['model_dir']
pretrained_model=os.listdir(dir_pretrained_models)

# pretrained_model=pretrained_model[0:2]
# # print(pretrained_model)
# edsr=pretrained_model[0]
# espcn= pretrained_model[1]


def get_upscaled_images_act1(img_small, filemodel_filepath, modelname, scale):
    model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
    print("Reading model file {}".format(filemodel_filepath))

    # setting up the model initialization
    model_pretrained.readModel(filemodel_filepath)
    model_pretrained.setModel(modelname, scale)

    # prediction or upscaling
    img_upscaled = model_pretrained.upsample(img_small)
    return img_upscaled


def design_upscale_action1(img_small,config):
    img_upscaled1 = get_upscaled_images(img_small, config['edsr_model'], "edsr", 4)
    img_upscaled2 = get_upscaled_images(img_small, config['espcn_model'], "espcn", 4)

    model_out ={"edsr_out":img_upscaled1, "espcn_out":img_upscaled2}

    shape= print(img_upscaled1.shape, img_upscaled2.shape)
    return model_out
def save_img_act1(out):
    cv2.imwrite('output/action_1_output/image_edsr.png', out['edsr_out'])
    cv2.imwrite('output/action_1_output/image_espcn.png', out['espcn_out'])

    #cv2.imwrite('output/image_sr.png', img5)
    msg= print("pretrained model output were saved")
    return msg



