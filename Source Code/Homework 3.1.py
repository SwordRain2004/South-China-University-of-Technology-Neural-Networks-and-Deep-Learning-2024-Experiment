import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 1、模型架构
# ① 编码器（全连接层）： 输入图片维度：784 (28 × 28) 隐藏层维度（ReLU）：256 输出层维度（Tanh）：512
# ② 生成均值（全连接层）： 输入层维度：512 输出层维度：2
# ③ 生成标准差（全连接层）： 输入层维度：512 输出层维度：2
# ④ 使用均值和标准差生成隐变量z
# ⑤ 解码器（全连接层）： 输入维度：2 隐藏层维度（ReLU）：512输出层维度（Sigmoid）：784
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2_mean = nn.Linear(256, 2)
        self.fc2_logvar = nn.Linear(256, 2)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mean + epsilon * std
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(2, 512)
        self.fc4 = nn.Linear(512, 784)
    def forward(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, logvar
def loss_function(reconstructed_x, x, mean, logvar):
    BCE = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
recon_losses = []
kl_losses = []
for epoch in range(num_epochs):
    model.train()
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_recon_loss += nn.functional.binary_cross_entropy(recon_batch, data, reduction='sum').item()
        train_kl_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    recon_losses.append(train_recon_loss / len(train_loader.dataset))
    kl_losses.append(train_kl_loss / len(train_loader.dataset))
# 训练完网络，需要提交重构损失和KL散度的随迭代次数的变化图，以及10 张生成的手写数字图片。
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), recon_losses, label='Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Reconstruction Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), kl_losses, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('KL Divergence vs. Epoch')
plt.legend()
plt.grid(True)
# plt.savefig('homework_3_1.png')
plt.show()
model.eval()
with torch.no_grad():
    z = torch.randn(10, 2).to(device)
    sample = model.decoder(z).cpu()
sample = sample.view(-1, 28, 28)
fig, axes = plt.subplots(1, 10, figsize=(10, 1))
for i in range(10):
    axes[i].imshow(sample[i].squeeze(), cmap='gray')
    axes[i].axis('off')
# plt.savefig('generated_digits.png')
plt.show()