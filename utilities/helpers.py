import os
import cv2
import numpy as np


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
    img_upscaled5 = get_upscaled_images(img_small, config['weighted_model'], "sr", 4)

    shape= print(img_upscaled1.shape, img_upscaled2.shape, img_upscaled3.shape, img_upscaled4.shape)
    return img_upscaled1,img_upscaled2,img_upscaled3,img_upscaled4,img_upscaled5




def plot_images(images, titles):
    fig = plt.figure(figsize=(20., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.1)

    i = 0
    for ax, img in zip(grid, images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[i])
        i += 1
    plt.show()


def save_img(img1,img2,img3,img4,img5):
    cv2.imwrite('output/image_edsr.png', img1)
    cv2.imwrite('output/image_espcn.png', img2)
    cv2.imwrite('output/image_fsrcnn.png', img3)
    cv2.imwrite('output/image_lapsrn.png', img4)
    cv2.imwrite('output/image_sr.png', img5)
    msg= print("pretrained model output were saved")
    return msg