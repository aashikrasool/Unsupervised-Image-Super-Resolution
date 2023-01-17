from PIL import Image, ImageEnhance
img="D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\image_edsr.png"
def increase_sharp(img):
    img = Image.open(img)
    enhancer_object = ImageEnhance.Sharpness(img)
    sharpened_img = enhancer_object.enhance(1.08)
    file_path = "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\output\\action6_out\\increased_brightened.png"
    sharpened_img.save(file_path)
    msg = print("image saved")
    return msg
increase_sharp(img)


