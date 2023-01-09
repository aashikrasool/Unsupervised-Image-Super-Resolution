from gaussian import *
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


img,img_small=downsample(img, scale=0.3, plot=False)
comb = comb_img()



