# _*_coding:utf-8_*_
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa

class AugmentationTransform:
    def __init__(self, config, augment=False): 
        self.size = config.size
        self.resize = getattr(config, "resize", None)
        self.augment_config = config.data_augment if augment else {}
        self.transformations = self._initialize_transforms()

    def _initialize_transforms(self):
        aug_list = []

        if self.size and self.resize:
            self.scale = iaa.Resize({"height": self.size[0], "width": self.size[1]}, interpolation="linear")
        else:
            self.scale = None

        if self.augment_config.get("GaussianBlur"):
            radius = random.random() * 1.2
            aug_list.append(iaa.GaussianBlur(sigma=radius))

        if self.augment_config.get("Gaussion_noise"):
            aug_list.append(iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255), per_channel=0.2))

        if self.augment_config.get("Sharpen"):
            aug_list.append(iaa.Sharpen(alpha=(0.0, 0.15), lightness=(0.8, 1.2)))

        if self.augment_config.get("ContrastNormalization"):
            aug_list.append(iaa.ContrastNormalization(alpha=(0.5, 1.5)))

        scale = (0.95, 1.05) if self.augment_config.get("AffineScale") else 1.0
        translate_percent = random.uniform(-0.05, 0.05) if self.augment_config.get("AffineTranslate") else 0
        rotate = random.uniform(-15, 15) if self.augment_config.get("AffineRotate") else 0
        shear = random.uniform(-5, 5) if self.augment_config.get("AffineShear") else 0

        if any([self.augment_config.get(key) for key in ["AffineScale", "AffineTranslate", "AffineRotate", "AffineShear"]]):
            aug_list.append(iaa.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate, shear=shear, mode='constant'))

        if self.augment_config.get("PiecewiseAffine"):
            aug_list.append(iaa.PiecewiseAffine(scale=(0.0, 0.04), nb_rows=(2, 4), nb_cols=(2, 4), mode='constant'))

        if self.augment_config.get("Fliplr"):
            aug_list.append(iaa.Fliplr(p=1))

        if self.augment_config.get("Flipud"):
            aug_list.append(iaa.Flipud(p=1))

        if self.augment_config.get("Multiply"):
            aug_list.append(iaa.Multiply(mul=(0.8, 1.2)))

        if self.augment_config.get("Dropout"):
            aug_list.append(iaa.Dropout(p=(0.0, 0.1)))

        return aug_list

    def __call__(self, image):
        image = np.expand_dims(image, 0)
        if self.scale:
            image = self.scale.augment_images(image)
        if self.transformations:
            seq = iaa.SomeOf((1, min(len(self.transformations) // 3 + 1, 3)), self.transformations)
            image = seq.augment_images(image)
        return np.squeeze(image, 0)
    
class ImageDataset(Dataset):
    def __init__(self, config, dataset_info, img_dir, label_file, transform=None):
        self.config = config
        self.dataset_info = dataset_info
        self.img_dir = img_dir
        self.transform = transform
        self.imgs = self._load_images(label_file)
        self.mean, self.std = self._set_normalization_params()

    def _load_images(self, label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()
        imgs = [(line.split()[1], int(line.split()[2])) if len(line.split()) == 3 else (line.split()[0], int(line.split()[1])) for line in lines]
        if self.config.debug:
            random.shuffle(imgs)
            return imgs[:6000]
        return imgs

    def _set_normalization_params(self):
        if self.dataset_info["modal"] == "rgb":
            return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif self.dataset_info["modal"] == "ir":
            return [0.5, 0.5, 0.5], [1, 1, 1]
        return [0, 0, 0], [1, 1, 1]

    def _normalize(self, img):
        return transforms.Normalize(self.mean, self.std)(img)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        if "HOG" in self.dataset_info["modal"]:
            img = cv2.imread(os.path.join(self.img_dir, img_path), 2).astype(np.float32)
            img = np.stack([img, img, img], 2)
        else:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_path)), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        img_tensor = transforms.ToTensor()(img).to(dtype=self.config.dtype)
        img_tensor = self._normalize(img_tensor)
        label_tensor = torch.tensor(label)
        return {'images': img_tensor, 'labels': label_tensor, 'items': os.path.splitext(img_path)[0]}

    def __len__(self):
        return len(self.imgs)
