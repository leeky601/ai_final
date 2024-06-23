from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
from torchvision import transforms
from model import SimpleCNN

app = Flask(__name__)

# 저장된 모델 로드
model = SimpleCNN(num_classes=15)  # 클래스 수를 맞춰 설정
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# 라벨 매핑 딕셔너리
label_map = {
    0: "통화중",
    1: "박수를 치는중",
    2: "자전거를 타는중",
    3: "춤을 추는중",
    4: "무언가를 마심",
    5: "먹는중",
    6: "싸우는 중",
    7: "포옹을 함",
    8: "웃고 있는 중",
    9: "감상",
    10: "달리는 중",
    11: "앉아 있는 중",
    12: "잠",
    13: "핸드폰을 하는 중",
    14: "노트북을 사용 중"
}

# 이미지 전처리 함수
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# 예측 함수
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return label_map[predicted_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_label = get_prediction(img_bytes)
        return jsonify({'class_label': class_label})

if __name__ == '__main__':
    app.run(debug=True)
