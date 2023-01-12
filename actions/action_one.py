import os,cv2
from utilities.helpers import *
def sample_action1(in_img, config):
    print("here",in_img)
    img, img_small = downsample(in_img, scale=0.4, plot=False)
    dir_pretrained_models = config['model_dir']
    os.listdir(dir_pretrained_models)
    # img_upscaled=get_upscaled_images(img_small, filemodel_filepath, modelname, scale)
    out_imgs=design_upscale(img_small)
    save_img(out_imgs)

    action_input_dataset = config['action1_input']
    out_file_img = [os.path.join(action_input_dataset, x) for x in os.listdir(dir_dataset)]
    input_img1 = cv2.imread(out_file_img[0], 1)
    input_img2 = cv2.imread(out_file_img[1], 1)


    comb = comb_img(input_img1,input_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return