#The options for the dataset augmentation.

from pythonUtils import print_Extension
import config
import torch

class ConfigOptions:
    def __init__(self, config):
        self.write = True
        self.num_workers = 6
        self.dataset_mode = config.Dataset_Mode
        self.size = config.Size
        self.resize = config.Resize
        self.datasets_list = config.DatasetsList
        self.dataset = config.Dataset
        self.data_augment = {
            "GaussianBlur": config.AugmentGaussianBlur,
            "GaussianNoise": config.AugmentGaussion_noise,
            "Sharpen": config.AugmentSharpen,
            "ContrastNormalization": config.AugmentContrastNormalization,
            "AffineScale": config.AugmentAffineScale,
            "AffineTranslate": config.AugmentAffineTranslate,
            "AffineRotate": config.AugmentAffineRotate,
            "AffineShear": config.AugmentAffineShear,
            "PiecewiseAffine": config.AugmentPiecewiseAffine,
            "Fliplr": config.AugmentFliplr,
            "Flipud": config.AugmentFlipud,
            "Dropout": config.AugmentDropout,
            "Multiply": config.AugmentMultiply,
            "Brightness": config.AugmentBrightness,
            "Saturate": config.AugmentSaturate,
        }
        self.inchannel = config.InChannel
        self.dtype = torch.float32

    def display_info(self, file=None):
        for key, value in self.__dict__.items():
            formatted_key = (str(key).capitalize() + ":").center(30, " ")
            print_Extension(f"{formatted_key}{value}", _file=file)
