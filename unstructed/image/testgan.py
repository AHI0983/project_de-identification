import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm  # 학습 진행 상태를 가시적으로 표시
from PIL import Image
import os

# 데이터셋 로드 및 전처리
class DogDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 이미지 전처리 정의 (GAN 학습을 위해 64x64로 resize하고, 정규화)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 픽셀 값을 -1에서 1 사이로 정규화
])

# Generator 정의 (GAN)
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

# Discriminator 정의 (GAN)
class Discriminator(nn.Module):
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

# GAN 학습 및 생성기 저장
def train_gan(epochs=100, batch_size=64, latent_size=100, train_dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, real_images in progress_bar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            labels_real = torch.full((batch_size,), 1.0, device=device)
            labels_fake = torch.full((batch_size,), 0.0, device=device)
            
            # Discriminator 학습 (real images)
            netD.zero_grad()
            output_real = netD(real_images).view(-1)
            loss_real = criterion(output_real, labels_real)
            loss_real.backward()

            # Fake images 생성
            noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach()).view(-1)
            loss_fake = criterion(output_fake, labels_fake)
            loss_fake.backward()
            optimizerD.step()

            # Generator 학습
            netG.zero_grad()
            output_fake = netD(fake_images).view(-1)
            lossG = criterion(output_fake, labels_real)
            lossG.backward()
            optimizerG.step()

            # tqdm을 통해 손실 정보 표시
            progress_bar.set_postfix({
                'Loss_D': (loss_real + loss_fake).item(),
                'Loss_G': lossG.item()
            })

        # 매 10 에포크마다 이미지 저장
        if epoch % 10 == 0:
            save_image(fake_images.data[:25], f"epoch_{epoch}.png", nrow=5, normalize=True)

    # 모델 저장
    torch.save(netG.state_dict(), "generator.pth")
    print("GAN 학습 완료")


# 훈련 이미지 경로 및 검증 이미지 경로 정의
train_img_dir = 'C:\\Users\\dkflt\\OneDrive\\문서\\카카오톡 받은 파일\\sample_code1\\images\\train'  # 실제 경로로 수정
val_img_dir = 'C:\\Users\\dkflt\\OneDrive\\문서\\카카오톡 받은 파일\\sample_code1\\val_images'      # 검증 경로도 동일하게 수정

img_files = os.listdir(train_img_dir)
print(f"Found {len(img_files)} image files in directory: {train_img_dir}")
print("Sample files:", img_files[:5])  # 첫 5개 파일 출력

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)
    print(f"Directory {train_img_dir} created.")
    


# 데이터셋 정의 (훈련 데이터셋)
train_dataset = DogDataset(img_dir=train_img_dir, transform=transform)

# 데이터셋 크기 확인 (훈련 데이터셋)
print(f"Train dataset size: {len(train_dataset)}")

# DataLoader 정의 (훈련 데이터셋)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 검증 데이터셋 정의
val_dataset = DogDataset(img_dir=val_img_dir, transform=transform)

# 데이터셋 크기 확인 (검증 데이터셋)
print(f"Val dataset size: {len(val_dataset)}")

# DataLoader 정의 (검증 데이터셋)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# GAN 학습 실행
train_gan(epochs=100, batch_size=64, latent_size=100, train_dataloader=train_dataloader)
