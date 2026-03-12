# Vision Transformer - Tibame 作業筆記 (Teaching Enhanced Version)

## 簡介
這是基於 Vision-Transformer-Tibame-HW.py 的改寫版本，提供更清楚的註解與架構，幫助各位理解及進行作業。

## 匯入必要的套件

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 設定超參數

```python
# 設定訓練��超參數
learning_rate = 0.001
num_epochs = 10
batch_size = 32
```

## 定義 Vision Transformer 模型
這邊定義了 Vision Transformer 的結構，細節請參考原始碼。

```python
class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # 定義模型中的各層
        self.encoder = nn.TransformerEncoder(...)  # 詳細結構根據需要進行設定

    def forward(self, x):
        # 注意：x 的形狀 [batch_size, num_channels, height, width]
        x = x.flatten(2)  # 扁平化處理
        x = self.encoder(x)
        
        # 備註：此行為作業中需要修改的地方
        x = torch.mean(x[:,1:], dim=1)  # 計算平均，移除第一個維度
        return x
```

## 訓練模型

```python
# 主訓練迴圈
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 將資料放入模型中進行訓練
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 結論
這份筆記是為了讓各位同學能更清楚理解 Vision Transformer 的運作邏輯與基本結構，希望能在作業中提供幫助。