import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def showSampleImage(img_path, window_name='sample image'):
    img = cv2.imread(img_path, 1)
    cv2.imshow(window_name, img)

def downsample(img_file, scale=0.3, plot=False):
    img = cv2.imread(img_file, 1)
    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    if plot:
        img_small_resize = cv2.resize(img_small, (img.shape[0], img.shape[1]))
        cv2.imshow("images window", np.hstack([img, img_small_resize]))
    return img, img_small

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

def design_upscale(img_small,config):
    img_upscaled1 = get_upscaled_images(img_small, config['edsr_model'], "edsr", 4)
    img_upscaled2 = get_upscaled_images(img_small, config['espcn_model'], "espcn", 4)
    img_upscaled3 = get_upscaled_images(img_small, config['fsrcnn_model'], "fsrcnn", 4)
    img_upscaled4 = get_upscaled_images(img_small, config['lapsrn_model'], "lapsrn", 4)
    #img_upscaled5 = get_upscaled_images(img_small, config['weighted_model'], "sr", 4)

    model_out ={"edsr_out":img_upscaled1, "espcn_out":img_upscaled2, "fsrcnn_out": img_upscaled3, "lapsrn_out":img_upscaled4}

    shape= print(img_upscaled1.shape, img_upscaled2.shape, img_upscaled3.shape, img_upscaled4.shape)
    return model_out




def plot_images(images, titles):
    fig = plt.figure(figsize=(20., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.1)

    i = 0
    for ax, img in zip(grid, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[i])
        i += 1
    plt.show()


def save_img(out):
    cv2.imwrite('input/action1/image_edsr.png', out['edsr_out'])
    cv2.imwrite('input/action1/image_espcn.png', out['espcn_out'])
    cv2.imwrite('input/action3/image_fsrcnn.png', out['fsrcnn_out'])
    cv2.imwrite('input/action3/image_lapsrn.png', out['lapsrn_out'])
    #cv2.imwrite('output/image_sr.png', img5)
    msg= print("pretrained model output were saved")
    return msg