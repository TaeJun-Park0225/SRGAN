# SRGAN

A Tenserflow implementation of SRGAN based on CVPR 2017 paper Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.

The train and val datasets are sampled from DIV2K. Train dataset has 800 images and tested with Set 5 dataset.

Enviroment

    python          3.9.13
    tensorflow      2.11.0
    numpy           1.21.5
    opencv-python   4.6.0.66
    
# Usage


## Train 

Put train image dataset in ./train folder.

You can find useful dataset on [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data)

To train dataset type code below on your commend.

    python train.py

Or you can choose directory to train using arguments(--train_dir).

optional arguments:

    --epochs                 epochs
    --batchs                 batchs
    --lr_g                   learning rate of generator
    --lr_d                   learning rate of discriminator
    --train_dir              directory of image to train / 학습 할 이미지 위치
    --load_model             load saved model / 저장된 모델 불러오기 (1: True, 0: False)
    --use_cpu                forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)

## Test Single Image

Put test image in ./test folder.

To train dataset type code below on your commend.

    python test.py

The result will be saved in ./result folder.

optional arguments:

    --target_folder         directory of image to process super resolution / super resolution 처리할 이미지 위치
    --save_folder           directory to save super resoultion image / super resolution 처리된 이미지 저장 위치
