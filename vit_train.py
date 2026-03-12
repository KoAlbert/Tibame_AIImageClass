"""
模型訓練腳本，針對CIFAR100資料集。
模組用途：透過Vision Transformer進行訓練。
包含數據集轉換、嵌入、訓練循環等主要功能。  
"""
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
from torchvision.models import VisionTransformer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 數據集轉換和正規化
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 數據集載入
full_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 主要模型架構
class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        # 塞入自定義參數
        self.model = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            num_classes=100,
            dim=embed_dim,
            depth=layers,
            heads=heads,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
        ).to(DEVICE)

    def forward(self, x):
        return self.model(x)

model = ViT().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 訓練循環
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

    # 驗證階段
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT Training for CIFAR100')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=32, help='Image size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--layers', type=int, default=6, help='Number of layers')
    args = parser.parse_args() 
    main()