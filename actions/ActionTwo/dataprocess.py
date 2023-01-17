import PIL
from PIL import Image
import os

#This code is just for making  LR images but we didn't use that

# fetch configuration
parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="testing configuration file")
args = parser.parse_args()
config = get_config(args.config)
project_root = os.getcwd()
train_dir = config['./HR/']
lr_path=config['./LR/']



#This code is just for making  LR images but we didn't use that

def resize(im,new_width):
  width,height= im.size
  ratio =height/width
  new_h= int(ratio*new_width)
  resized_Image= im.resize((new_width,new_h))
  return resized_Image

from IPython.core import extensions
files= os.listdir(train_dir)
extensions= ['jpg','jpeg','png','gif','.png']
for file in files:
  ext= file.split(".")[-1]
  #print(ext)
  if ext in extensions:
    im= Image.open(train_dir+"\\"+file)
    im_resized= resize(im,96)
    filepath= f(lr_path)
    im_resized.save(filepath)