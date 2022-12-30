A Tenserflow implementation of SRGAN based on CVPR 2017 paper Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.

The train and val datasets are sampled from DIV2K. Train dataset has 800 images and tested with Set 5 dataset.

Usage

Train 
python train.py

optional arguments:
--epochs                 epochs
--batchs                 batchs
--lr_g                   learning rate of generator
--lr_d'                  learning rate of discriminator
--train_dir              directory of image to train / 학습 할 이미지 위치
--load_model             load saved model / 저장된 모델 불러오기 (1: True, 0: False)
--use_cpu                forced to use CPU only / CPU 만 이용해 학습하기 (1: True, 0: False)

Test Single Image
python test.py

optional arguments:
--target_folder         directory of image to process super resolution / super resolution 처리할 이미지 위치
--save_folder           directory to save super resoultion image / super resolution 처리된 이미지 저장 위치
