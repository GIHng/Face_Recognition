import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # 첫 번째 합성곱 레이어
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        
        # 세 번째 합성곱 레이어
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2)
        
        # 전체 연결 레이어
        self.fc1 = nn.Linear(64 * 12 * 12, 256)  # MaxPool 3번 적용 후 이미지 크기가 12x12가 됩니다.

        # Dropout 추가 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 첫 번째 합성곱 레이어와 활성화 함수, 풀링 적용
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # 두 번째 합성곱 레이어와 활성화 함수, 풀링 적용
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # 세 번째 합성곱 레이어와 활성화 함수, 풀링 적용
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        # 전체 연결 레이어를 위해 Flatten
        x = x.view(x.size(0), -1)
        
        # 전체 연결 레이어 적용
        x = self.fc1(x)
        
        # add Dropout layer 
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
