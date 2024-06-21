from pythonUtils import log_message
import configuration as cfg
import torch

class Settings:
    def __init__(self, config: cfg):
        # Tensorboard monitoring
        self.enable_logging = True
        
        # Dataset preparation configurations
        self.worker_count = 6
        self.dataset_type = config.Dataset_Mode
        self.image_size = config.Size
        self.image_resize = config.Resize
        self.dataset_info = config.Dataset
        self.class_count = None
        self.data_augmentation = {
            "GaussianBlur": config.AugmentGaussianBlur,
            "GaussianNoise": config.AugmentGaussion_noise,
            "Sharpen": config.AugmentSharpen,
            "ContrastNormalization": config.AugmentContrastNormalization,
            "AffineScale": config.AugmentAffineScale,
            "AffineTranslate": config.AugmentAffineTranslate,
            "AffineRotate": config.AugmentAffineRotate,
            "AffineShear": config.AugmentAffineShear,
            "PiecewiseAffine": config.AugmentPiecewiseAffine,
            "FlipHorizontal": config.AugmentFliplr,
            "FlipVertical": config.AugmentFlipud,
            "Multiply": config.AugmentMultiply,
            "Dropout": config.AugmentDropout,
        }

        # Model configurations
        self.model_categories = config.Model_Group
        self.use_pretrained = config.Pretrain
        self.pretrained_path = config.Pretrain_pth
        
        # Optimizer configurations
        self.learning_rate_milestones = config.Lr_Decay_Milestones
        self.learning_rate_gamma = config.Lr_Decay_Gamma
        self.learning_rate_lambda = config.Lr_decay_Lambda
        self.learning_rate = config.LR

        # Training configurations
        self.epoch_count = config.Epochs
        self.batch_size = config.BatchSize
        self.test_split_percentage = config.TestPercent
        self.test_frequency = config.TestRate
        self.loss_function = config.Loss
        self.mse_mode = config.MSEmode
        
        # Device and hardware configurations
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_channels = config.InChannel
        self.use_multi_gpu = config.Mul_Gpu_Train
        self.parameter_initialization = config.ParamInit
        self.data_type = torch.float32

    def display_info(self, file=None):
        for attribute, value in self.__dict__.items():
            formatted_key = f"{str(attribute).replace('_', ' ').capitalize()}:".center(30, " ")
            log_message(formatted_key + str(value), _file=file)
