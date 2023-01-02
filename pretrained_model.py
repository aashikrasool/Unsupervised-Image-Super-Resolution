import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# Removed platform-dependent code
project_root = os.getcwd()
dir_dataset = os.path.join(project_root, "HR")
files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]


img = cv2.imread(files_img[8], 1)
window_name = 'image sample'
cv2.imshow(window_name,img)
# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)




def downsample(img_file, scale=0.3, plot=False):
    img = cv2.imread(img_file, 1)
    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if plot:
        img_small_resize = cv2.resize(img_small, (img.shape[0], img.shape[1]))
        cv2.imshow("images window", np.hstack([img, img_small_resize]))
    return img, img_small




_, img_small = downsample(files_img[8], scale=0.4, plot=True)

# !rm -rf pretrained_models
# !wget https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb -P pretrained_models -q
# !wget https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb -P pretrained_models -q
# !wget https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -P pretrained_models -q
# !wget https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb -P pretrained_models -q

dir_pretrained_models = os.path.join(project_root, "pretrained_models")
os.listdir(dir_pretrained_models)


# ''' Model upscale any image using opencv and external pretrained models. '''
def get_upscaled_images(img_small, filemodel_filepath, modelname, scale):
    model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
    print("Reading model file {}".format(filemodel_filepath))

    # setting up the model initialization
    model_pretrained.readModel(filemodel_filepath)
    model_pretrained.setModel(modelname, scale)

    # prediction or upscaling
    img_upscaled = model_pretrained.upsample(img_small)
    return img_upscaled


img, img_small = downsample(files_img[8], scale=0.25)
print(img.shape, img_small.shape)
img_upscaled1 = get_upscaled_images(img_small, os.path.join(dir_pretrained_models,"EDSR_x4.pb"), "edsr", 4)
img_upscaled2 = get_upscaled_images(img_small, os.path.join(dir_pretrained_models,"ESPCN_x4.pb"), "espcn", 4)
img_upscaled3 = get_upscaled_images(img_small, os.path.join(dir_pretrained_models,"FSRCNN_x4.pb"), "fsrcnn", 4)
img_upscaled4 = get_upscaled_images(img_small, os.path.join(dir_pretrained_models,"LapSRN_x4.pb"), "lapsrn", 4)

print(img_upscaled1.shape, img_upscaled2.shape, img_upscaled3.shape, img_upscaled4.shape)



def plot_images(images, titles):
    fig = plt.figure(figsize=(20., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.1)

    i = 0
    for ax, img in zip(grid, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[i])
        i += 1
    plt.show()

img_small_resize = cv2.resize(img_small, (img.shape[0], img.shape[1]))

titles = ["original", "downsampled", "edsr", "espcn", "fsrcnn", "lapsrn"]
images = [img, img_small_resize, img_upscaled1, img_upscaled2, img_upscaled3, img_upscaled4]
plot_images(images, titles)