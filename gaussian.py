#first we will design gaussian pyramid
import cv2
import numpy as np
input_img1 = cv2.imread("C:\\Users\\aashi\\Unsupervised-Image-Super-Resolution\\HR\\0.png")

layer_first = input_img1.copy()
#gaussian pyramid
gaussian_pyramid = [layer_first]
for i in range (3):
    layer_first = cv2.pyrDown(layer_first)
    gaussian_pyramid.append(layer_first)
#laplacian pyramid
layer_first= gaussian_pyramid[2]
laplacian_py = [layer_first]
for i in range(2,0,-1):
    size = (gaussian_pyramid[i-1].shape[1],gaussian_pyramid[i-1].shape[0])
    gaussian_expand = cv2.pyrUp(gaussian_pyramid[i], dstsize=size)
    laplacian = cv2.subtract(gaussian_pyramid[i-1],gaussian_expand)
    laplacian_py.append(laplacian)
    cv2.imshow(str(i), laplacian)

reconstruct_image = laplacian_py[0]
for i in range(1,3):
    size = (laplacian_py[i].shape[1], laplacian_py[i].shape[0])
    reconstruct_image = cv2.pyrUp(reconstruct_image, dstsize=size)
    reconstruct_image = cv2.add(reconstruct_image, laplacian_py[i])
    cv2.imshow(str(i), reconstruct_image)

# cv2.imshow("img1",gaussian_pyramid[0])
# cv2.imshow("img2",gaussian_pyramid[1])
# cv2.imshow("img3",gaussian_pyramid[2])
cv2.imshow("original image", input_img1)

cv2.waitKey(0)
cv2.destroyAllWindows()