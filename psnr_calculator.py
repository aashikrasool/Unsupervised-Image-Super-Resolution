import cv2
import os
import numpy as np
from skimage.measure import compare_ssim

# project_root = os.getcwd()
# dir_dataset = os.path.join(project_root, "HR")
# files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]

img1 = cv2.imread('action_1_output/weighted3.png')
img2 = cv2.imread('HR/16.png')

def psnr(img1,img2):
    psnr = cv2.PSNR(img1, img2)
    psnr=print("PSNR :",psnr)
    return psnr


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err=print("MSE:",err)
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def ssim(imag1,img2):
    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    sim=print("SSIM: {}".format(score))
    return sim




mse(img1,img2)
psnr(img1,img2)
ssim(img1,img2)

# print(psnr)