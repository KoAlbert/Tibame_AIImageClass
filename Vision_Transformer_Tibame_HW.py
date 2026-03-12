import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, random_split
import argparse

# 超參數設定
LEARNING_RATE = 0.001  # 學習率
WEIGHT_DECAY = 0.0001  # 權重衰減
BATCH_SIZE = 32         # 批量大小
NUM_EPOCHS = 10        # 訓練次數
IMAGE_SIZE = 72        # 圖片大小
PATCH_SIZE = 6         # 補丁大小
PROJECTION_DIM = 64    # 投影維度
NUM_HEADS = 4          # 注意力頭數
TRANSFORMER_LAYERS = 8  # 變壓器層數
MLP_HEAD_UNITS = [2048, 1024]  # MLP 超參數

# 計算 CIFAR100 訓練集的均值和標準差
def compute_mean_std(dataset):
    mean = 0.0
    std = 0.0
    for images, _ in dataset:
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])
    mean /= len(dataset)
    std /= len(dataset)
    return mean, std

# 建立訓練、驗證和測試的 DataLoader
def create_dataloaders(batch_size, image_size):
    # 設定資料增強和轉換
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 重新調整大小
        transforms.RandomRotation(7),                  # 隨機旋轉
        transforms.RandomHorizontalFlip(0.5),         # 隨機水平翻轉
        transforms.ToTensor(),                        # 轉換為張量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 標準化
    ])
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 重新調整大小
        transforms.ToTensor(),                        # 轉換為張量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),   # 標準化
    ])

    # 下載 CIFAR100 數據集
    dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    mean, std = compute_mean_std(dataset)

    # 訓練集和驗證集的劃分
    valid_size = int(0.3 * len(dataset))
    train_size = len(dataset) - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, mean, std

# 定義模型的各個組件
class CreatePatchesLayer(nn.Module):
    # 定義生成補丁的層
    pass  # 完成這部分代碼

class PatchEmbeddingLayer(nn.Module):
    # 定義補丁嵌入的層
    pass  # 完成這部分代碼

class TransformerBlock(nn.Module):
    # 定義變壓器區塊
    pass  # 完成這部分代碼

def create_mlp_block(input_dim, mlp_units):
    # 定義 MLP 區塊
    pass  # 完成這部分代碼

class ViTClassifierModel(nn.Module):
    # 定義 ViT 分類器模型
    def __init__(self):
        super(ViTClassifierModel, self).__init__()
        # 初始化模型的各個層
        pass  # 完成這部分代碼

    def forward(self, x):
        # 定義前向傳播
        return torch.mean(x[:, 1:], dim=1)

# 計算準確度函數
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels.data).item() / len(labels)

# 計算 top-5 準確度
def calculate_accuracy_top_5(outputs, labels):
    top5_preds = outputs.topk(5, dim=1)[1]
    correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
    return correct.sum().item() / labels.size(0)

# 訓練網絡
def train_network(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()  # 將梯度歸零
        outputs = model(images)  # 前向傳播
        loss = criterion(outputs, labels)  # 計算損失
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 測試網絡
def test_network(model, valid_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            correct += calculate_accuracy(outputs, labels)
            total += len(labels)
    return correct / total

# 主函數
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training and validation')
    args = parser.parse_args()

    train_loader, valid_loader, mean, std = create_dataloaders(args.batch_size, IMAGE_SIZE)
    model = ViTClassifierModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_network(model, train_loader, criterion, optimizer)
        valid_accuracy = test_network(model, valid_loader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')  # 打印訓練和驗證準確率

if __name__ == '__main__':
    main()