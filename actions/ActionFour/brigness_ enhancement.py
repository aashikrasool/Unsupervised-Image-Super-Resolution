
from PIL import Image
from PIL import ImageEnhance
from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/config.yaml', help="testing configuration file")
# args = parser.parse_args()
# config = get_config(args.config)

img="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\image_edsr.png"

def increase_bright(img):
    img = Image.open(img)
    # Removed platform-dependent code
    # project_root = os.getcwd()
    # dir_dataset = config['data_dir']
    # files_img = [os.path.join(dir_dataset, x) for x in os.listdir(dir_dataset)]
    # files_img=files_img[8]
    curr_bri = ImageEnhance.Brightness(img)
    new_bri = 1.08
    brightened = curr_bri.enhance(new_bri)
    file_path = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action4_output\\increased_brightened.png"
    brightened.save(file_path)
    msg=print("image saved")
    return msg

increase_bright(img)