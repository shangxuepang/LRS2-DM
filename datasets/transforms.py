from torchvision import transforms
import random
from PIL import Image, ImageEnhance, ImageOps

class RandomApplyEnhance:
    def __init__(self, brightness=0.2, contrast=0.2, sharpness=0.2, color=0.2, probability=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.color = color
        self.prob = probability

    def __call__(self, img):
        if random.random() < self.prob:
            img = ImageEnhance.Brightness(img).enhance(1 + random.uniform(-self.brightness, self.brightness))
        if random.random() < self.prob:
            img = ImageEnhance.Contrast(img).enhance(1 + random.uniform(-self.contrast, self.contrast))
        if random.random() < self.prob:
            img = ImageEnhance.Sharpness(img).enhance(1 + random.uniform(-self.sharpness, self.sharpness))
        if random.random() < self.prob:
            img = ImageEnhance.Color(img).enhance(1 + random.uniform(-self.color, self.color))
        return img

class RandomGaussianBlur:
    def __init__(self, probability=0.3, radius_range=(0.1, 1.5)):
        self.prob = probability
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomSolarize:
    def __init__(self, probability=0.2, threshold=192):
        self.prob = probability
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.prob:
            return ImageOps.solarize(img, self.threshold)
        return img


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((560, 560)),
            transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            RandomApplyEnhance(0.3, 0.3, 0.3, 0.2, 0.7),
            RandomGaussianBlur(probability=0.3),
            RandomSolarize(probability=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
