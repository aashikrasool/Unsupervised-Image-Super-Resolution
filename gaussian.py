#first we will design gaussian pyramid
import cv2
import numpy as np
import os

project_root = os.getcwd()
dir_dataset = os.path.join(project_root, "HR")
file_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
input_img1 = cv2.imread(file_img[1], 1)
input_img2 = cv2.imread(file_img[6], 1)

#downsampled  output1
first_layer= cv2.pyrDown(input_img1)
second_layer= cv2.pyrDown(first_layer)

#for 2 image
first_layer2 = cv2.pyrDown(input_img2)
second_layer2= cv2.pyrDown(first_layer2)

#laplacian pyramid
#expanding image
expand_image_first_l=  cv2.pyrUp(first_layer)
expand_image_second_l=  cv2.pyrUp(second_layer)

#expanding image2
expand_image_first_l2=  cv2.pyrUp(first_layer2)
expand_image_second_l2=  cv2.pyrUp(second_layer2)

#laplacian
laplacian11 = cv2.subtract(input_img1, expand_image_first_l)
laplacian12 = cv2.subtract(first_layer, expand_image_second_l)

#laplacian 2
laplacian21 = cv2.subtract(input_img2, expand_image_first_l2)
laplacian22 = cv2.subtract(first_layer2, expand_image_second_l2)
#if in case to ccheck the size of images
#but this dataset has same size images(384,384)
#print(laplacian22.shape)

final_lap1= cv2.add(laplacian11,laplacian21)
final_lap2 =cv2.add(laplacian12,laplacian22)
#we need this image to reconstruct  the image
final_gauss=  cv2.add(second_layer,second_layer2)

# #to reconstruct image
reconstruct1=cv2.pyrUp(laplacian11)
reconstruct2=cv2.pyrUp(laplacian12)
reconstruct3= cv2.pyrUp(laplacian21)
reconstruct4 = cv2.pyrUp(laplacian22)

reconstruct = [reconstruct1, reconstruct2, reconstruct3,reconstruct4]
#laplacian


cv2.imshow("second",reconstruct4)
cv2.imshow("second",reconstruct3)

#display expanded img
# cv2.imshow("exp1",expand_image_first_l)

#

#to display  images
#cv2.imshow("first downsampled",first_layer)
#cv2.imshow("second downsampled",second_layer)
# cv2.imshow("Ground Truth",input_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()