import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from utilities.manage_config import *
from utilities.gaussian import *
from utilities.metrics_calculator import *
from utilities.helpers import *
from argparse import ArgumentParser
from actions.action_one import *

# Main pipeline file

# fetch configuration
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="testing configuration file")
args = parser.parse_args()
config = get_config(args.config)



# Removed platform-dependent code
project_root = os.getcwd()
dir_dataset = config['data_dir']
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]


showSampleImage(files_img[8])


# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)


_, img_small = downsample(files_img[8], scale=0.4, plot=True)

# !rm -rf pretrained_models
# !wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb -P pretrained_models -q
# !wget https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb -P pretrained_models -q
# !wget https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -P pretrained_models -q
# !wget https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb -P pretrained_models -q

dir_pretrained_models = config['model_dir']
os.listdir(dir_pretrained_models)


# downsample an image and get the original image, feed to the network and get output



img, img_small = downsample(files_img[8], scale=0.25)
print(img.shape, img_small.shape)


img_small_resize = cv2.resize(img_small, (img.shape[0], img.shape[1]))

out_imgs = design_upscale(img_small,config)
save_img(out_imgs)

titles = ["original", "downsampled", "edsr", "espcn", "fsrcnn", "lapsrn"]
images = [img, img_small_resize, out_imgs['edsr_out'], out_imgs['espcn_out'], out_imgs['fsrcnn_out'], out_imgs['lapsrn_out']]
plot_images(images, titles)




#action 1
img = cv2.imread(files_img[8], 1)
#sample_action1(img, config)