import cv2
import numpy as np
import os
from utilities.manage_config import get_config

config = get_config("./configs/config.yaml")
project_root = os.getcwd()
dir_dataset = config['input/action3']

file_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
input_img1 = cv2.imread(file_img[2], 1)
input_img2 = cv2.imread(file_img[3], 1)
def action3(img1,img2):
    comb = cv2.addWeighted(img1, 0.6,img2, 0.4, 0.0)
    cv2.imwrite('action_1_output/action3.png', comb)
    comb=print(cv2.imshow("comb",comb))
    return comb
action3(img2,img1)

