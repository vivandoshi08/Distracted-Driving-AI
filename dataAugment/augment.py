# Augment the Data
import os
import sys
from threading import Thread, Lock
from imgaug import augmenters as iaa
import numpy as np
import cv2
import random
from tqdm import tqdm

sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from pythonUtils import *

class ImageAugmentor:
    def __init__(self, options, augment=False):
        self.size = options.size
        self.resize = getattr(options, "resize", None)
        self.augment_settings = options.data_augment if augment else {}

        self.transforms = []
        self.scale = self._initialize_scale()
        self._initialize_augmentations()

    def _initialize_scale(self):
        if self.size and self.resize:
            return iaa.Resize({"height": self.size[0], "width": self.size[1]}, interpolation="linear")
        return None

    def _initialize_augmentations(self):
        augmenters = {
            "GaussianBlur": lambda: iaa.GaussianBlur(sigma=random.random() * 1.2),
            "GaussianNoise": lambda: iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255), per_channel=0.2),
            "Sharpen": lambda: iaa.Sharpen(alpha=(0.0, 0.15), lightness=(0.8, 1.2)),
            "ContrastNormalization": lambda: iaa.ContrastNormalization(alpha=(0.5, 1.5)),
            "Affine": self._initialize_affine,
            "PiecewiseAffine": lambda: iaa.PiecewiseAffine(scale=(0.0, 0.04), nb_rows=(2, 4), nb_cols=(2, 4)),
            "Fliplr": lambda: iaa.Fliplr(p=1),
            "Flipud": lambda: iaa.Flipud(p=1),
            "Multiply": lambda: iaa.Multiply(mul=(0.8, 1.2)),
            "Dropout": lambda: iaa.Dropout(p=(0.0, 0.1)),
            "Brightness": lambda: iaa.imgcorruptlike.Brightness(severity=random.randint(1, 5)),
            "Saturate": lambda: iaa.imgcorruptlike.Saturate(severity=random.randint(1, 5))
        }

        for augment, func in augmenters.items():
            if self.augment_settings.get(augment, False):
                if augment == "Affine":
                    self.transforms.extend(func())
                else:
                    self.transforms.append(func())

    def _initialize_affine(self):
        scale = (0.95, 1.05) if self.augment_settings.get("AffineScale", False) else 1.0
        translate_percent = random.uniform(-1, 1) * 0.05 if self.augment_settings.get("AffineTranslate", False) else 0
        rotate = random.uniform(-1, 1) * 25 if self.augment_settings.get("AffineRotate", False) else 0
        shear = random.uniform(-1, 1) * 5 if self.augment_settings.get("AffineShear", False) else 0

        return [iaa.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate, shear=shear, mode='constant')]

    def __call__(self, image):
        image = np.expand_dims(image, 0)
        if self.scale:
            image = self.scale.augment_images(image)
        if self.transforms:
            seq = iaa.SomeOf((1, min(len(self.transforms) // 3 + 1, 3)), self.transforms, random_order=True)
            image = seq.augment_images(image)
        return np.squeeze(image, 0)


def process_images_mt(image_root, image_dest, images, file_handle, thread_id, augmenter, count=6, lock=Lock()):
    for img_path in tqdm(images, position=0, desc=f'Thread:{thread_id+1}'):
        path, label = img_path.split()
        img = cv2.imread(os.path.join(image_root, path))
        if augmenter.scale and not augmenter.transforms:
            augmented_img = augmenter(img)
            save_image(os.path.join(image_dest, path), augmented_img, file_handle, path, label, lock)
        else:
            for i in range(count):
                augmented_img = augmenter(img)
                save_image(os.path.join(image_dest, f"{path}_{i}"), augmented_img, file_handle, f"{path}_{i}", label, lock)
    return


def process_images(image_root, image_dest, images, file_handle, augmenter, count=6):
    for img_path in tqdm(images, position=0, desc='Thread:1'):
        path, label = img_path.split()
        img = cv2.imread(os.path.join(image_root, path))
        if augmenter.scale and not augmenter.transforms:
            save_image(os.path.join(image_dest, path), img, file_handle, path, label)
        else:
            for i in range(count):
                augmented_img = augmenter(img)
                save_image(os.path.join(image_dest, f"{path}_{i}"), augmented_img, file_handle, f"{path}_{i}", label)
    return


def save_image(save_path, image, file_handle, path, label, lock=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not os.path.exists(save_path):
        try:
            cv2.imwrite(save_path, image)
        except Exception as e:
            print(f"Error saving {save_path}: {e}")
        if lock:
            lock.acquire()
        file_handle.write(f"{path} {label}\n")
        if lock:
            lock.release()
    return


if __name__ == '__main__':
    import distutils.util
    import argparse
    import options
    import config
    from config import *
    
    parser = argparse.ArgumentParser(description='Augmentation')
    parser.add_argument('--dataset', default='None', type=str, help='train dataset')
    parser.add_argument('--subset', default='None', type=str, help='train dataset subset')
    parser.add_argument('--augment', default=False, type=lambda x: bool(distutils.util.strtobool(x)), help='augment dataset')
    args = parser.parse_args()
    print(args.augment)

    project = ProjectInfo()
    opt = options.Options(config)
    augmentor = ImageAugmentor(opt, args.augment)

    subset = args.subset if args.subset != 'None' else "Test"
    if args.dataset != 'None':
        opt.dataset = eval(args.dataset)
    
    image_root = opt.dataset[f"Image{subset}Path"]
    train_label_path = os.path.join(project.ROOT, opt.dataset[f"{subset}LabelPath"])
    new_label_path = train_label_path.replace('.txt', '_augment.txt' if args.augment else '_size224.txt')
    image_dest = f"{image_root}_Augment" if args.augment else f"{image_root}_size224"

    with open(train_label_path, 'r') as label_file:
        labels = [line.strip().split() for line in label_file]
        images = [f"{image} {gt}" for idx, image, gt in (labels if len(labels[0]) == 3 else [(None, *lbl) for lbl in labels])]

    lock = Lock()
    threads = []
    image_batches = np.array_split(images, 10)

    with open(new_label_path, 'w') as new_label_file:
        for i, batch in enumerate(image_batches):
            thread = Thread(target=process_images_mt, args=(image_root, image_dest, batch, new_label_file, i, augmentor, 11, lock))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    print('Main thread finished...')
