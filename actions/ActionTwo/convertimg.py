import cv2
import tensorflow as tf
img=cv2.imread("D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action2_output\\newo_image.png",1)
org_image=cv2.imread("D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR\\8.png",1)
psnr = tf.image.psnr(
    tf.clip_by_value(img, 0, 255),
    tf.clip_by_value(org_image, 0, 255), max_val=255)
print("PSNR Achieved: %f" % psnr)
# Above are just for checking
# print(cv2.imshow("comb",img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()