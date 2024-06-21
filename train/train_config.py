from pythonUtils import *
import sys
sys.path.append('labelfolder')
import datasetDriver100

dataset = datasetDriver100.Driver100_Cross_Camera_Setting_D1

epochs = 50
batch_size = 64
test_percent = 0
test_rate = 5
loss_function = None
mse_mode = ""
lr_decay_milestones = [40, 45, 50]
lr_decay_gamma = 0.1
lr_decay_lambda = None
learning_rate = 0.001

device_type = "cuda"
use_pretrain = False
pretrain_path = "./pretrainFactory/SFD(trainsub)&AUCv1/epoch_100.pth"

model_group = ['MobileNetV3_Small_Pretrain_ScoreLoss', 'Resnet18_Pretrain_ScoreLoss'][:2]
model_group = [i for i in model_group if "#" not in i]
dataset_mode = ["RGB"]
input_channels = 3
input_size = (224, 224)
resize_images = True
multi_gpu_training = False
parameter_initialization = True

augment_gaussian_blur = False
augment_gaussian_noise = False
augment_sharpen = False
augment_contrast_normalization = False
augment_affine_scale = False
augment_affine_translate = False
augment_affine_rotate = False
augment_affine_shear = False
augment_piecewise_affine = False
augment_flip_lr = False
augment_flip_ud = False
augment_multiply = False
augment_dropout = False
augment_brightness = False
augment_saturate = False
