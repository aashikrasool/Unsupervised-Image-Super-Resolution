import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# project_root = os.getcwd()
# dir_dataset = os.path.join(project_root, "HR")
# files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]

img1 = cv2.imread('action_1_output/weighted32.png')
img2 = cv2.imread('HR/16.png')

def psnr(img1,img2):
    psnr = cv2.PSNR(img1, img2)
    psnr=print("PSNR :",psnr)
    return psnr

def mse(img1, img2):
   heigt=384
   width=384
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(heigt*width))
   mse= print("MSE  :",mse)
   return mse

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


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    ssim_map=print("SSIM :",ssim_map.mean())
    return ssim


# def ssim(imag1,img2):
#     ysim= ssim(imag1,img2)
#     sim=print("SSIM: ",ysim)
#     return sim
#


mse(img1,img2)
psnr(img1,img2)
ssim(img1,img2)

# print(psnr)