import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
from facenet_pytorch import MTCNN  # MTCNN 얼굴 탐지기

# MTCNN 모델 초기화
mtcnn = MTCNN(keep_all=True)

# 이미지 전처리 설정 (GAN에서 학습한 크기로 변환)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # GAN에서 학습한 이미지와 일치하게 정규화
])

# Generator 모델 클래스 정의 (학습된 GAN 모델 사용)
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

# MTCNN을 사용한 얼굴 탐지 함수
def detect_faces(image_path):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"파일 경로가 존재하지 않습니다: {image_path}")
        
        image = Image.open(image_path)
        boxes, _ = mtcnn.detect(image)
        
        if boxes is None:
            print(f"얼굴을 찾지 못했습니다: {image_path}")
            return None, []

        # 얼굴 영역을 반환
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # 좌표값으로 변환
            w, h = x2 - x1, y2 - y1
            faces.append((x1, y1, w, h))

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), faces

    except FileNotFoundError as fnf_error:
        print(f"파일 경로 오류: {fnf_error}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
    
    return None, []

# 얼굴에 GAN을 적용하는 함수
def apply_gan_to_faces(generator, image, faces, device, latent_size=100):
    generator.eval()
    with torch.no_grad():
        for (x, y, w, h) in faces:
            # 얼굴 영역을 추출
            face_region = image[y:y+h, x:x+w]
            face_image = Image.fromarray(face_region)
            
            # 얼굴 영역을 GAN의 입력 크기로 변환
            face_tensor = transform(face_image).unsqueeze(0).to(device)
            
            # GAN으로 가짜 얼굴 생성
            noise = torch.randn(1, latent_size, 1, 1, device=device)
            fake_face = generator(noise)
            fake_face = fake_face.squeeze().permute(1, 2, 0).cpu().numpy()

            # 생성된 가짜 얼굴을 원래 얼굴 크기로 변환
            transformed_face = ((fake_face + 1) * 127.5).astype('uint8')

            # 생성된 얼굴과 원본 얼굴 크기가 다를 경우 크기 조정
            transformed_face = cv2.resize(transformed_face, (w, h))

            # 크기 불일치를 해결하기 위해 맞추는 부분
            transformed_face_height, transformed_face_width, _ = transformed_face.shape
            if transformed_face_height != h or transformed_face_width != w:
                print(f"크기 불일치 발생: 생성된 얼굴 크기 ({transformed_face_width}, {transformed_face_height}) != 원본 크기 ({w}, {h})")
                transformed_face = cv2.resize(transformed_face, (w, h))

            # 원본 이미지에 가짜 얼굴 대체
            image[y:y+h, x:x+w] = transformed_face

    return image

# 비식별화 함수
def anonymize_faces_in_image(generator, image_path, device):
    image, faces = detect_faces(image_path)
    
    if image is None:
        print(f"이미지를 처리할 수 없습니다: {image_path}")
        return

    if len(faces) > 0:
        transformed_image = apply_gan_to_faces(generator, image, faces, device)
        
        # 파일 이름 변경 후 저장
        base_name = os.path.basename(image_path)
        folder_path = os.path.dirname(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(folder_path, f"{name}-deiden{ext}")
        
        cv2.imwrite(output_path, transformed_image)
        print(f"비식별화된 이미지 저장 완료: {output_path}")
    else:
        print(f"얼굴을 찾지 못했기 때문에 이미지를 저장하지 않습니다: {image_path}")

# 폴더 내 모든 이미지를 처리하는 함수
def process_images_in_folder(folder_path, generator, device):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, file_name)
            anonymize_faces_in_image(generator, image_path, device)

# GAN 모델과 이미지를 불러와 처리하는 메인 함수
def main(input_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator().to(device)
    
    # 학습된 GAN 모델 가중치 불러오기
    netG.load_state_dict(torch.load("generator.pth"))
    print("GAN 모델 가중치 불러오기 완료")
    
    if os.path.isdir(input_path):
        # 폴더 내 모든 이미지 처리
        process_images_in_folder(input_path, netG, device)
    else:
        # 단일 파일 처리
        anonymize_faces_in_image(netG, input_path, device)

if __name__ == "__main__":
    input_path = input("비식별화할 이미지 또는 폴더 경로를 입력하세요: ")
    main(input_path)
