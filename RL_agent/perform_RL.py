import gym
from gym import Env
from gym.spaces import Discrete,Box,Dict,Tuple,MultiBinary,MultiDiscrete
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import cv2

'''--------------------------------------------------------------------------------------------------------------
--------------------------------Building own env--------------------------------------------------------------------
--building agent to solve our issue
'''
img = cv2.imread("G:\\Gachon Masters\\pycharm\\reinforcement\\0upsampled.jpg")
class SuperImg(Env):

    def __init__(self):

        self.action_space=Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(384,384,3), dtype=np.uint8)
        self.state=cv2.imread("G:\\Gachon Masters\\pycharm\\reinforcement\\0upsampled.jpg")
        self.img=cv2.imread("G:\\Gachon Masters\\pycharm\\reinforcement\\0upsampled.jpg")
        self.models = {
            0: self._create_action1(self.state),
            1: self._create_action2(self.state),
            2: self._create_action3(self.state)
        }
        self.process_length= 120

    def step(self,action):
        super_res_img = self.models[action](self.state)
        psnr = self._calculate_psnr(super_res_img)
        self.process_length -= 1
        if self.psnr >35 and self.psnr <=49:
            reward=1
        else:
            reward=-1

        if self.process_length <= 0:
            done=True
        else:
            done=False
        info = {}

        return self.state,reward,done,info

    def get_upscaled_img(self,img_small, model_path, modelname, scale):
        model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
        model_pretrained.readModel(model_path)
        model_pretrained.setModel(modelname, scale)
        img_upscaled = model_pretrained.upsample(img_small)
        return img_upscaled

    def per_matrix(self,img1, img2):
        # MSE
        heigt = 384
        width = 384
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff ** 2)
        mse = err / (float(heigt * width))

    def downsample(self,img_file, scale=0.3, plot=False):
        img = cv2.imread(img_file, 1)
        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return img, img_small

    def _create_action1(self,img):
        img, img_small = self.downsample(img, 0.4)
        # img_g = "C:\\Users\\aashi\\PycharmProjects\\Unsupervised-Image-Super-Resolution\\HR\\0.png"
        modelName1, model_path1, _, _, _, _, scale = self.model_detail()
        up_img = self.get_upscaled_img(img_small, model_path1, modelName1, scale)
        # psnr = per_matrix(up_img, img_g)
        return up_img

    def build_hr(self,model_path, scale, modelname1, img_small):
        model_pretrained = cv2.dnn_superres.DnnSuperResImpl_create()
        print("Reading model file {}".format(model_path))
        model_pretrained.readModel(model_path)
        model1 = model_pretrained.setModel(modelname1, scale)

        img_upscaled1 = model1.upsample(img_small)

        return img_upscaled1

    def model_detail(self):
        model_path1 = "C:\\Users\\aashi\\PycharmProjects\\Unsupervised-Image-Super-Resolution\\pretrained_models\\EDSR_x4.pb"
        model_path2 = "C:\\Users\\aashi\\PycharmProjects\\Unsupervised-Image-Super-Resolution\\pretrained_models\\ESPCN_x4.pb"
        model_path3 = "C:\\Users\\aashi\\PycharmProjects\\Unsupervised-Image-Super-Resolution\\pretrained_models\\FSRCNN_x4.pb"
        modelName1 = "edsr"
        modelName2 = "espcn"
        modelName3 = "fsrcnn"
        scale = 4
        return modelName1, model_path1, modelName2, model_path2, modelName3, model_path3, scale

    def action2(self,img):
        img, img_small = self.downsample(img, 0.4)
        _, _, modelName2, model_path2, _, _, scale = self.model_detail()
        up_img = self.get_upscaled_img(img_small, model_path2, modelName2, scale)

        return up_img

    def action3(self,img):
        img, img_small = self.downsample(img, 0.4)
        _, _, _, _, modelName3, model_path3, scale = self.model_detail()
        up_img = self.get_upscaled_img(img_small, model_path3, modelName3, scale)
        return up_img

    def render(self):
        pass
    def reset(self):
        # Reset the state of the environment
        self.state = self.img
        self.process_length=120

        return self.state

#print(gym.spaces.Box(low=0, high=255, shape=(3,384,384,), dtype=np.uint8))
env= SuperImg()
env.observation_space.sample()