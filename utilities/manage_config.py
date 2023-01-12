import os
import time
import shutil
import yaml
import io
import wget


# import cv2
# import numpy as np

####################################################################
####################################################################
# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def print_config(config):
    # print('#'*80)
    print('project configuration:')
    with open(config, "r") as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def write_config(config, yaml_path):
    # Write YAML file
    yaml_data = []
    with open(config, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with io.open(yaml_path, 'w', encoding='utf8') as outfile:
        yaml.dump(yaml_data, outfile, default_flow_style=False, allow_unicode=True)

def get_subdir_name(dir_path):
    sub_dir_list = []
    for dir_name in os.listdir(dir_path):
        sub_dir_list.append(dir_name)
    return sub_dir_list


def read_lines(input_txtFile):
    with open(input_txtFile, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def download_pretrained(config):
    output_dir = config['model_dir']
    _ = wget.download("https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb", out=output_dir)
    _ = wget.download("https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb", out=output_dir)
    _ = wget.download("https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb", out=output_dir)
    _ = wget.download("https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb", out=output_dir)


