import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # GAN에서 학습된 크기와 동일하게 변환
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize([0.5], [0.5])  # 픽셀 값을 [-1, 1]로 정규화
])

# Haar Cascades 경로 설정
cascades_path = r'C:\dog\haarcascade_frontalface_default.xml'  # 파일 경로 수정

# 얼굴 인식을 위한 Haar Cascades 모델 불러오기
face_cascade = cv2.CascadeClassifier(cascades_path)

if face_cascade.empty():
    raise FileNotFoundError(f"Haar Cascade XML 파일을 불러올 수 없습니다: {cascades_path}")

# Generator 클래스 정의 (학습된 가중치를 불러오기 위해 필요)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 얼굴 탐지 함수
def detect_faces(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 탐지 성공률을 높이기 위해 scaleFactor와 minNeighbors 조정
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        return image, faces
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None, []

# 얼굴에만 GAN 적용 함수
def apply_gan_to_faces(generator, image, faces, device, latent_size=100):
    generator.eval()
    with torch.no_grad():
        for (x, y, w, h) in faces:
            face_region = image[y:y+h, x:x+w]
            face_image = Image.fromarray(face_region)
            face_tensor = transform(face_image).unsqueeze(0).to(device)
            
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            fake_face = generator(noise)
            fake_face = fake_face.squeeze().permute(1, 2, 0).cpu().numpy()
            
            transformed_face = ((fake_face + 1) * 127.5).astype('uint8')
            transformed_face = cv2.resize(transformed_face, (w, h))
            image[y:y+h, x:x+w] = transformed_face

    return image

# 얼굴 비식별화 적용 함수
def anonymize_faces_in_image(generator, image_path, device):
    image, faces = detect_faces(image_path)
    
    if image is None:
        print(f"이미지를 처리할 수 없습니다: {image_path}")
        return

    if len(faces) > 0:
        transformed_image = apply_gan_to_faces(generator, image, faces, device)
        
        # 파일 이름 변경: 원본 파일 이름 뒤에 '-deiden' 붙이기
        base_name = os.path.basename(image_path)
        folder_path = os.path.dirname(image_path)  # 원본 폴더 경로 가져오기
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(folder_path, f"{name}-deiden{ext}")
        
        cv2.imwrite(output_path, transformed_image)
        print(f"비식별화된 이미지 저장 완료: {output_path}")
    else:
        print(f"얼굴을 찾지 못했습니다: {image_path}")

# 폴더 내 모든 이미지를 처리하는 함수
def process_images_in_folder(folder_path, generator, device):
    for file_name in os.listdir(folder_path):
        # 이미지 파일만 처리
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, file_name)
            anonymize_faces_in_image(generator, image_path, device)

# 학습된 모델 및 이미지를 로드
def main(input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("generator.pth"))  # 학습된 모델을 불러오기

    # 입력이 폴더인지 파일인지 확인
    if os.path.isdir(input_path):
        # 폴더 내 모든 이미지를 처리
        process_images_in_folder(input_path, netG, device)
    else:
        # 단일 파일 처리
        anonymize_faces_in_image(netG, input_path, device)

if __name__ == "__main__":
    input_path = input("비식별화할 이미지 또는 폴더 경로를 입력하세요: ")
    main(input_path)
