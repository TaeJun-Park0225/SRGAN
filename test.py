import tensorflow as tf
import argparse
from glob import glob
import numpy as np
import cv2
import os

#load trained model / 학습된 이미지 불러오기
Generator = tf.keras.models.load_model("Generator.h5")
Generator.summary()

parser = argparse.ArgumentParser()
parser.add_argument('--target_folder', required=False, default="./test/", help='directory of image to process super resolution / super resolution 처리할 이미지 위치')
parser.add_argument('--save_folder', required=False, default="./SR_result/", help='directory to save super resoultion image / super resolution 처리된 이미지 저장 위치')
args = parser.parse_args()
target_folder, save_folder = args.target_folder, args.save_folder

img_dirs = glob(target_folder + "/*")
for i in range(len(img_dirs)):

    # Read image / 이미지 읽어오기
    img_dir = img_dirs[i]
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255

    # Process super resoultion / 이미지에 super resoultion 적용
    img_sr = Generator.predict(np.expand_dims(img, 0))
    img_sr = np.array(img_sr[0] * 127.5 + 127.5)
    img_sr[img_sr >= 255] = 255
    img_sr[img_sr <= 1] = 1
    img_sr = np.array(img_sr, dtype=np.uint8)
    img_sr = cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR)

    # Save processed image / 이미지 저장
    save_dir = save_folder + os.path.split(img_dir)[-1]
    cv2.imwrite(save_dir, img_sr)
