from utilities.gaussfsian import *
from pretrained_model import *
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


project_root = os.getcwd()
dir_dataset = os.path.join(project_root, "HR")
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
img = cv2.imread(files_img[8], 1)

# window_name = 'image sample'
# cv2.imshow(window_name,img)
# cv2.waitKey(0)


img,img_small=downsample(img, scale=0.4, plot=False)
dir_pretrained_models = os.path.join(project_root, "pretrained_models")
os.listdir(dir_pretrained_models)
# img_upscaled=get_upscaled_images(img_small, filemodel_filepath, modelname, scale)
(img_upscaled1,img_upscaled2,img_upscaled3,img_upscaled4)=design_upscale(img_small)
save_img(img_upscaled1,img_upscaled2,img_upscaled3,img_upscaled4)

action_input_dataset = os.path.join(project_root, "input_img_action1")
out_file_img = [os.path.join(action_input_dataset, x) for x in os.listdir(dir_dataset)]
input_img1 = cv2.imread(out_file_img[0], 1)
input_img2 = cv2.imread(out_file_img[1], 1)
input_img3= cv2.imread(out_file_img[2],1)


comb = comb_img(input_img1,input_img2,input_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

