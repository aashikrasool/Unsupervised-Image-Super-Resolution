import PIL
from PIL import Image
import os

train_dir= "D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR"

def resize(im,new_width):
  width,height= im.size
  ratio =height/width
  new_h= int(ratio*new_width)
  resized_Image= im.resize((new_width,new_h))
  return resized_Image

from IPython.core import extensions
files= os.listdir("D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\HR")
extensions= ['jpg','jpeg','png','gif','.png']
for file in files:
  ext= file.split(".")[-1]
  #print(ext)
  if ext in extensions:
    im= Image.open(train_dir+"\\"+file)
    im_resized= resize(im,96)
    filepath= f"D:\\research_pycharm\\reisr\\Unsupervised-Image-Super-Resolution\\LR\\{file}"
    im_resized.save(filepath)