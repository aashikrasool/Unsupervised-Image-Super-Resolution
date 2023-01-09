import cv2
import os
# project_root = os.getcwd()
# dir_dataset = os.path.join(project_root, "HR")
# files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]

img1 = cv2.imread('action_1_output/weighted3.png')
img2 = cv2.imread('HR/16.png')

def psnr(img1,img2):
    psnr = cv2.PSNR(img1, img2)
    psnr=print("PSNR :",psnr)
    return psnr

psnr(img1,img2)

# print(psnr)