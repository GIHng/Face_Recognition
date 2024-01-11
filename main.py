import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.transform import get_augmented_transforms
from models.simple_CNN import SimpleCNN
from training.train import Train
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

image_size = 96
batch_size = 100
learning_rate = 0.0005
num_classes = 28
log_dir = 'logs/'

TRAINING_FILE = os.path.join(os.getcwd(),  "data/desc/training_file_list.txt")
VALIDATION_FILE = os.path.join(os.getcwd(), "data/desc/validate_file_list.txt")

print(TRAINING_FILE)


def main():
    # 모델 인스턴스 생성
    model = SimpleCNN(num_classes)
    train = Train(writer)

    # DataLoader 설정 및 데이터 로딩
    transform = get_augmented_transforms(image_size)

    train_dataset = CustomDataset(TRAINING_FILE, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(VALIDATION_FILE, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    train.train_model(model, train_loader, val_loader, 100, optimizer, criterion)

    writer.flush()

if __name__ == '__main__':
    main()
