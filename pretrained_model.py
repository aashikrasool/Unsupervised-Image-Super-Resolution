import cv2
import os

# assigning  data path
dir_dataset = ""
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
