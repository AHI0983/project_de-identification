import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset  # Dataset 추가
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os

# 이미지 전처리를 위한 transform 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # 이미지 크기를 64x64로 조정
    transforms.ToTensor(),         # 이미지를 Tensor로 변환
    transforms.Normalize([0.5], [0.5])  # 픽셀 값을 -1에서 1 사이로 정규화
])

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
# 데이터셋 로드 (강아지 이미지 경로 설정)
dataset = CustomImageDataset(img_dir=r'C:\dog\edog', transform=transform)

# 데이터셋 크기 출력
print(f"Dataset size: {len(dataset)} images")

# DataLoader 생성
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# GAN 관련 코드 (Generator, Discriminator 및 학습 코드)

class Generator(nn.Module):
    # 생성자 네트워크 정의
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


class Discriminator(nn.Module):
    # 판별자 네트워크 정의
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

# 손실 함수 및 최적화 도구
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0003, betas=(0.5, 0.999))

# 학습 루프 (tqdm으로 진행 상황 표시)
latent_size = 100
num_epochs = 200

for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, data in enumerate(progress_bar, 0):
        real_images = data.to(device)
        batch_size = real_images.size(0)
        
        # 레이블 정의
        labels_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)
        labels_fake = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        # 판별자 학습
        netD.zero_grad()
        output_real = netD(real_images).view(-1)
        loss_real = criterion(output_real, labels_real)
        loss_real.backward()

        # 가짜 이미지 생성 및 판별자 학습
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = netG(noise)
        output_fake = netD(fake_images.detach()).view(-1)
        loss_fake = criterion(output_fake, labels_fake)
        loss_fake.backward()
        optimizerD.step()

        # 생성자 학습
        netG.zero_grad()
        output_fake = netD(fake_images).view(-1)
        lossG = criterion(output_fake, labels_real)
        lossG.backward()
        optimizerG.step()

        # tqdm의 진행 상황에 손실 정보 표시
        progress_bar.set_postfix({
            'Loss_D': (loss_real + loss_fake).item(),
            'Loss_G': lossG.item()
        })
        
    # 학습 결과 확인을 위한 샘플 이미지 저장
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_images = netG(noise).detach().cpu()
            save_image(fake_images, f'epoch_{epoch}.png', normalize=True)

# 학습 완료 후 Generator 모델 저장
torch.save(netG.state_dict(), "generator.pth")  # Generator의 가중치 저장
print("GAN 학습 완료!")
