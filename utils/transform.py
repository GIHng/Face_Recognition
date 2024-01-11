from PIL import Image, ImageDraw
from torchvision import transforms    
import numpy as np

RAMDOM_ROTATION = 10
RANDOM_PROB = 0.5

def add_similar_pattern(image):

    # 어둡게 변경
    pattern = Image.new('RGB', image.size, (55, 55, 55))
    
    # 패턴 이미지에 사각형 그리기 (빨간색)
    draw = ImageDraw.Draw(pattern)

    # 상대 좌표를 사용하여 사각형의 위치와 크기 지정
    width, height = image.size
    left = int(0 * width)    
    top = int(0.25 * height)    
    right = int(1 * width)   
    bottom = int(0.75 * height)
    
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255))
        
    # 원본 이미지와 패턴 이미지를 합성 (반투명)
    return Image.blend(image, pattern, alpha=0.7)


def add_watermark(image):
    watermark = Image.new('RGB', image.size, (55, 55, 55))
    draw = ImageDraw.Draw(watermark)
    width, height = image.size

    draw.text((0 * width, 0.5 * height), "WATERMARK", fill=(255, 255, 255))
    
    watermark = watermark.resize(image.size)
    
    return Image.blend(image, watermark, alpha=0.5)


def get_augmented_transforms(image_size):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(RAMDOM_ROTATION),
            transforms.Lambda(lambda img: add_watermark(img) if np.random.rand() < RANDOM_PROB else img),  # 50% 확률로 워터마크 추가
            transforms.Lambda(lambda img: add_similar_pattern(img) if np.random.rand() < RANDOM_PROB else img),  # 50% 확률로 유사한 패턴 추가
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return transform

def default_get_augmented_transforms(image_size):
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(RAMDOM_ROTATION),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    return transform