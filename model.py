import torch.nn as nn

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=15):  # 클래스 수를 인자로 받도록 수정
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*56*56, 128)  # 입력 크기를 반영하여 수정
        self.fc2 = nn.Linear(128, num_classes)  # 클래스 수 반영

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(kernel_size=2)(x)
        x = x.view(-1, 64*56*56)  # 입력 크기를 반영하여 수정
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
