import cv2
import os
import numpy as np

from skimage.metrics import structural_similarity
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

# project_root = os.getcwd()
# dir_dataset = os.path.join(project_root, "HR")
# files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]

img2 = cv2.imread('D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output\\converted\\new_image.png')
img1 = cv2.imread('HR/8.png')

def per_matrix(img1,img2):
    #MSE
    heigt=384
    width=384
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(heigt*width))


    #Psnr
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    else:
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        cpsnr= print("psnr :",psnr)
        cmse= print("MSE :",mse)
        return cmse,cpsnr

def ssim(imag1,img2):
    img1_gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    img2_gray= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    (score,diff)= structural_similarity(img1_gray,img2_gray,full = True)
    ssim= print("ssim:", score)
    return ssim


# def mse(imageA, imageB):
#     # the 'Mean Squared Error' between the two images is the
#     # sum of the squared difference between the two images;
#     # NOTE: the two images must have the same dimension
#     err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
#     err /= float(imageA.shape[0] * imageA.shape[1])
#     err=print("MSE:",err)
#     # return the MSE, the lower the error, the more "similar"
#     # the two images are
#     return err




per_matrix(img1,img2)
ssim(img1,img2)


# print(psnr)