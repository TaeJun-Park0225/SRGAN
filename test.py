import tensorflow as tf
import argparse
from glob import glob
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--target_folder', required=False, default="./Set5/LRbicx3/", help='directory of image to process super resolution')
parser.add_argument('--save_folder', required=False, default="./SR_result/", help='directory to save super resoultion image')
args = parser.parse_args()
target_folder, save_folder = args.target_folder, args.save_folder
img_dirs = glob(target_folder + "/*")
save_dir = save_folder

Generator = tf.keras.models.load_model("Generator.h5")

for i in range(len(img_dirs)):
    img_dir = img_dirs[i]
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img_sr = Generator.predict(np.expand_dims(img, 0))
    img_sr = np.array(img_sr[0] * 127.5 + 127.5)
    img_sr[img_sr >= 255] = 255
    img_sr[img_sr <= 1] = 1
    
    img_sr = np.array(img_sr, dtype=np.uint8)
    print(np.max(img_sr), np.min(img_sr))
    img_sr = cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR)
    save_dir = save_folder + os.path.split(img_dir)[-1]
    print(save_dir)
    cv2.imwrite(save_dir, img_sr)
