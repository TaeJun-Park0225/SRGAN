# SRGAN

A Tenserflow implementation of SRGAN based on CVPR 2017 paper Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.

The train and val datasets are sampled from DIV2K. Train dataset has 800 images and tested with Set 5 dataset.

#Usage

##Train 
python train.py

optional arguments:
--epochs                 epochs
--batchs                 batchs
--lr_g                   learning rate of generator
--lr_d'                  learning rate of discriminator
--train_dir              directory of image to train / 학습 할 이미지 위치
--load_model             load saved model / 저장된 모델 불러오기 (1: True, 0: False)
--use_cpu                forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)

##Test Single Image
python test.py

optional arguments:
--target_folder         directory of image to process super resolution / super resolution 처리할 이미지 위치
--save_folder           directory to save super resoultion image / super resolution 처리된 이미지 저장 위치

![image](https://user-images.githubusercontent.com/79778595/210081306-be5392f8-353f-41c6-9d84-a4716b7e21d3.png)
![image](https://user-images.githubusercontent.com/79778595/210081321-e55de866-b0ac-46d9-a66f-72cba553d292.png)
![image](https://user-images.githubusercontent.com/79778595/210081263-6a4c5a45-c629-4d94-9ebd-046d5052582e.png)
![image](https://user-images.githubusercontent.com/79778595/210081281-f3e4e7db-20f7-4d64-af62-6a84ed1203c3.png)
![image](https://user-images.githubusercontent.com/79778595/210081339-a6f30373-37eb-48fa-8447-739e51a0df64.png)

This result trained with very few images compate to other SRGAN.

Please train whith enough before use this model.

# Usage

## 1. Train 
python train.py

optional arguments

    --epochs          epochs
    --batchs          batchs
    --lr_g            learning rate of generator
    --lr_d            learning rate of discriminator
    --train_dir       directory of image to train / 학습 할 이미지 위치
    --load_model      load saved model / 저장된 모델 불러오기 (1: True, 0: False)
    --use_cpu         forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)

## 2. Test Single Image
python test.py

optional arguments

    --target_folder   directory of image to process super resolution / super resolution 처리할 이미지 위치
    --save_folder     directory to save super resoultion image / super resolution 처리된 이미지 저장 위치
