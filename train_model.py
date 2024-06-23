import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import io

# 데이터 로드
train_data = pd.read_parquet('data/train.parquet')
test_data = pd.read_parquet('data/test.parquet')

# 클래스 수 확인
num_classes = train_data['labels'].nunique()
print(f'Number of classes: {num_classes}')

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모든 이미지를 224x224 크기로 조정
    transforms.Grayscale(num_output_channels=1),  # 이미지를 흑백으로 변환
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# PyTorch Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data.iloc[idx]['image']['bytes']  # 이미지 데이터
        label = self.data.iloc[idx]['labels']  # 레이블 데이터

        # 바이트 형식을 PIL Image로 변환
        image = Image.open(io.BytesIO(image_data))

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터셋 생성
train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의 및 학습 설정
model = SimpleCNN(num_classes=num_classes)  # 클래스 수를 전달
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 모델 저장
torch.save(model.state_dict(), 'model.pth')
