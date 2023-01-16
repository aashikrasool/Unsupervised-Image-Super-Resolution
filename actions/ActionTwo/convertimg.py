
import cv2
import numpy as np
def convert(img1,file_path):
    im = cv2.imread(img1)

    im_8bit = np.uint8(im)
    img=cv2.imwrite(file_path, im_8bit)
    return img

def shape(img):
    im = cv2.imread(img)
    img=print("Image size:", im.shape)
    return img




im="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output\\converted\\new_image.png"
img="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\output\\action2_output\\ne_image.png"
file_path="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output\\converted\\new_image.png"
shape(im)