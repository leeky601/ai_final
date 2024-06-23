import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
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
    transforms.Grayscale(num_output_channels=3),  # 이미지를 3채널로 변환
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

# ResNet 모델 정의 및 학습 설정
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 조기 종료 설정
early_stopping_patience = 5
early_stopping_counter = 0
best_loss = float('inf')

# 모델 학습
num_epochs = 50  # 에포크 수 증가
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch + 1}/{num_epochs}')

    # 훈련 단계
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 검증 단계
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(test_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # 조기 종료 조건 체크
    if val_loss < best_loss:
        best_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

# 최종 모델 저장
torch.save(model.state_dict(), 'final_model.pth')
