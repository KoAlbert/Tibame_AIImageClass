# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) 從零開始訓練 CIFAR100 分類器
本腳本完整對應 notebook「Vision_Transformer_Tibame_HW」的副本.ipynb
備註等級 B：核心邏輯逐行說明；import 等簡單敘述不逐行展開
"""

import sys
from typing import Callable, Type

import numpy as np
import torch
import torchvision

# ── 超參數（完全對應 notebook 預設值）──────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有 GPU 就用 cuda:0，否則用 CPU
LEARNING_RATE = 0.001       # AdamW 學習率
WEIGHT_DECAY = 0.0001       # AdamW 權重衰減，抑制過擬合
BATCH_SIZE = 32             # 每批次樣本數；class token 參數的第一維與此一致，因此 drop_last=True 可防止最後批次大小不符
NUM_EPOCHS = 10             # 訓練週期數（notebook 原始預設值，實驗時可改成 100）
IMAGE_SIZE = 72             # 輸入影像重新縮放後的邊長（pixel）
PATCH_SIZE = 6              # 每個 patch 的邊長（pixel）；patch 面積 = PATCH_SIZE² = 36
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # patch 總數 = (72//6)² = 144
PROJECTION_DIM = 64         # 每個 patch 投影後的嵌入維度（embed_dim）
NUM_HEADS = 4               # Multi-Head Attention 的頭數
TRANSFORMER_LAYERS = 8      # 堆疊 TransformerBlock 的層數
MLP_HEAD_UNITS = [2048, 1024]  # 分類 MLP 隱藏層各層輸出維度


# ── 第一步：計算訓練集的逐通道 mean / std ──────────────────────────────────────
# 先用純 ToTensor 建立臨時 DataLoader，迭代一遍計算 mean 與 std
# 之後 Normalize 才能讓輸入分布接近標準常態，加速訓練收斂
_tmp_transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]  # 僅轉張量，不做其他前處理
)

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=_tmp_transforms
)  # 下載 CIFAR100 訓練集（50000 張，100 類）

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,   # 將資料預先釘在記憶體，加速 CPU→GPU 傳輸
    num_workers=4,     # 使用 4 個子程序並行載入資料
)

mean = torch.zeros(3).to(DEVICE)  # 初始化 RGB 三通道累計 mean 為 0
std = torch.zeros(3).to(DEVICE)   # 初始化 RGB 三通道累計 std 為 0

for batch in trainloader:
    image = batch[0].to(DEVICE)                         # image 形狀：(B, 3, H, W)
    image_mean = torch.mean(image, dim=(0, 2, 3))       # 對 batch、H、W 維度取平均，保留 (3,) 通道維
    image_std = torch.std(image, dim=(0, 2, 3))         # 同樣對空間與批次維度取標準差
    mean = torch.add(mean, image_mean)                  # 累加各批次的 mean
    std = torch.add(std, image_std)                     # 累加各批次的 std

mean = (mean / len(trainloader)).to("cpu")  # 除以批次數得到估計均值，移回 CPU 供 Normalize 使用
std = (std / len(trainloader)).to("cpu")    # 同上，移回 CPU
print(mean)
print(std)


# ── 第二步：建立正式 transforms（含 Normalize）────────────────────────────────
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 統一縮放至 72×72
        torchvision.transforms.RandomRotation(degrees=7),         # 隨機旋轉 ±7° 做資料增強
        torchvision.transforms.RandomHorizontalFlip(p=0.5),       # 50% 機率水平翻轉
        torchvision.transforms.ToTensor(),                        # 轉為 float tensor，值域 [0,1]
        torchvision.transforms.Normalize(mean, std),              # 依計算出的 mean/std 標準化
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 縮放至 72×72（與訓練一致）
        torchvision.transforms.ToTensor(),                        # 轉張量
        torchvision.transforms.Normalize(mean, std),              # 用相同 mean/std 標準化，確保評估穩定
    ]
)


# ── 第三步：建立 Dataset 與 DataLoader ─────────────────────────────────────────
train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=train_transforms
)  # 訓練集套用包含增強的 transforms

valid_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=test_transforms
)  # 測試集（共 10000 張）套用不含隨機增強的 transforms

# 按 notebook 設計：將原始測試集再切分成 70% 驗證、30% 最終測試
valid_set, test_set = torch.utils.data.random_split(
    valid_dataset,
    [0.7, 0.3],
    generator=torch.Generator().manual_seed(42),  # 固定亂數種子，確保切分可重現
)

# drop_last=True：捨棄不足一個 batch 的最後批次
# 原因：class_parameter 形狀為 (BATCH_SIZE, num_added_token, embed_dim)，
# 若最後批次樣本數 < BATCH_SIZE 則張量維度不符，拼接時會出錯
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)
validloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)


# ── 模型元件 ───────────────────────────────────────────────────────────────────

class CreatePatchesLayer(torch.nn.Module):
    """使用 Unfold 將影像切成不重疊的 patch 序列。"""

    def __init__(self, patch_size: int, strides: int) -> None:
        super().__init__()
        # torch.nn.Unfold 以滑動視窗方式擷取 patch
        # kernel_size=patch_size：每個視窗大小為 patch_size×patch_size
        # stride=strides：步幅等於 patch_size 代表不重疊切割
        self.unfold_layer = torch.nn.Unfold(kernel_size=patch_size, stride=strides)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images 形狀：(B, C, H, W) = (32, 3, 72, 72)
        patched_images = self.unfold_layer(images)
        # Unfold 輸出形狀：(B, C*P², N) = (32, 3*36, 144)，N = NUM_PATCHES
        return patched_images.permute((0, 2, 1))
        # permute 後形狀：(B, N, C*P²) = (32, 144, 108)，對應「序列長度、特徵維度」排列


class PatchEmbeddingLayer(torch.nn.Module):
    """將 patch 序列線性投影後，加入 class token 與位置嵌入。"""

    def __init__(
        self,
        num_patches: int,
        batch_size: int,
        patch_size: int,
        embed_dim: int,
        device: torch.device,
        num_added_token: int = 1,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_added_token = num_added_token
        self.device = device

        # 位置嵌入：對「patch 數量 + class token 數」建立可學習的嵌入表
        # 每個位置對應一個長度為 embed_dim 的向量，讓模型知道各 patch 的空間位置
        self.position_emb = torch.nn.Embedding(
            num_embeddings=num_patches + num_added_token,  # 144 + 1 = 145
            embedding_dim=embed_dim,                       # 64
        )

        # 線性投影層：將每個 patch 的原始像素展開向量投影到 embed_dim 空間
        # 輸入維度 = patch_size² * 3（RGB）= 6*6*3 = 108；輸出維度 = 64
        self.projection_layer = torch.nn.Linear(patch_size * patch_size * 3, embed_dim)

        # class token：可學習參數，形狀 (batch_size, num_added_token, embed_dim)
        # 附加在 patch 序列最前方，其對應的輸出表示整張影像的全域資訊
        # 注意：形狀的第一維綁定 batch_size，因此 DataLoader 需 drop_last=True
        self.class_parameter = torch.nn.Parameter(
            torch.rand(batch_size, num_added_token, embed_dim).to(device),
            requires_grad=True,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # 建立位置索引：[0, 1, 2, ..., num_patches]，形狀 (1, 145)
        positions = (
            torch.arange(start=0, end=self.num_patches + self.num_added_token, step=1)
            .to(self.device)
            .unsqueeze(dim=0)
        )

        # patches 形狀：(B, N, C*P²) = (32, 144, 108)
        patches = self.projection_layer(patches)
        # 投影後形狀：(B, N, embed_dim) = (32, 144, 64)

        # 在序列最前方拼接 class token，使序列長度從 144 變為 145
        encoded_patches = torch.cat((self.class_parameter, patches), dim=1)
        # encoded_patches 形狀：(B, 145, 64)

        # 加上位置嵌入（廣播對齊 batch 維度）
        encoded_patches = encoded_patches + self.position_emb(positions)
        # 輸出形狀仍為：(B, 145, 64)

        return encoded_patches


def create_mlp_block(
    input_features: int,
    output_features: list,
    activation_function: Type[torch.nn.Module],
    dropout_rate: float,
) -> torch.nn.Module:
    """建立 Feed-Forward Network（FFN）模組，每層包含 Linear → Activation → Dropout。"""
    layer_list = []
    for idx in range(len(output_features)):
        if idx == 0:
            # 第一層：input_features → output_features[0]
            linear_layer = torch.nn.Linear(
                in_features=input_features, out_features=output_features[idx]
            )
        else:
            # 後續層：output_features[idx-1] → output_features[idx]
            linear_layer = torch.nn.Linear(
                in_features=output_features[idx - 1], out_features=output_features[idx]
            )
        dropout = torch.nn.Dropout(p=dropout_rate)
        # 每個子層封裝成 Sequential：Linear → Activation → Dropout
        layers = torch.nn.Sequential(linear_layer, activation_function(), dropout)
        layer_list.append(layers)
    return torch.nn.Sequential(*layer_list)  # 串接所有子層


class TransformerBlock(torch.nn.Module):
    """Transformer Encoder Block：LayerNorm + Multi-Head Self-Attention + FFN + 殘差連接。"""

    def __init__(
        self,
        num_heads: int,     # 注意力頭數（4）
        key_dim: int,       # Key/Value 維度（與 embed_dim 相同，64）
        embed_dim: int,     # Token 嵌入維度（64）
        ff_dim: int,        # FFN 中間層維度（128 = PROJECTION_DIM * 2）
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        # 輸入 LayerNorm：在進入注意力之前對序列做正規化
        self.layer_norm_input = torch.nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

        # Multi-Head Self-Attention：batch_first=True 代表輸入形狀為 (B, L, D)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim,
            batch_first=True,
        )

        self.dropout_1 = torch.nn.Dropout(p=dropout_rate)

        # 第一個殘差後的 LayerNorm
        self.layer_norm_1 = torch.nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        # 第二個殘差後的 LayerNorm（FFN 後）
        self.layer_norm_2 = torch.nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)

        # FFN（Feed-Forward Network）：embed_dim → ff_dim → embed_dim，使用 GELU 激活
        self.ffn = create_mlp_block(
            input_features=embed_dim,
            output_features=[ff_dim, embed_dim],  # 中間層擴展再壓縮回 embed_dim
            activation_function=torch.nn.GELU,
            dropout_rate=dropout_rate,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs 形狀：(B, L, embed_dim) = (32, 145, 64)
        layer_norm_inputs = self.layer_norm_input(inputs)  # 前置 LayerNorm

        # Self-Attention：Q=K=V=layer_norm_inputs，輸出形狀相同
        attention_output, _ = self.attn(
            query=layer_norm_inputs,
            key=layer_norm_inputs,
            value=layer_norm_inputs,
        )
        attention_output = self.dropout_1(attention_output)  # Attention 後 Dropout

        # 第一個殘差連接：inputs + attention_output，再做 LayerNorm
        out1 = self.layer_norm_1(inputs + attention_output)

        # FFN 前向傳播
        ffn_output = self.ffn(out1)

        # 第二個殘差連接：out1 + ffn_output，再做 LayerNorm
        output = self.layer_norm_2(out1 + ffn_output)
        return output  # 輸出形狀：(B, 145, 64)


class ViTClassifierModel(torch.nn.Module):
    """完整 ViT 分類模型：Patch Embedding → Transformer × N → Mean Pooling → MLP Head。"""

    def __init__(
        self,
        num_transformer_layers: int,
        embed_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        patch_size: int,
        num_patches: int,
        mlp_head_units: list,
        num_classes: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        # 切割 patch 的 Unfold 層
        self.create_patch_layer = CreatePatchesLayer(patch_size, patch_size)

        # Patch Embedding：線性投影 + class token + 位置嵌入
        self.patch_embedding_layer = PatchEmbeddingLayer(
            num_patches, batch_size, patch_size, embed_dim, device
        )

        # 堆疊 num_transformer_layers 個 TransformerBlock（8 層）
        self.transformer_layers = torch.nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerBlock(num_heads, embed_dim, embed_dim, feed_forward_dim)
            )

        # 分類 MLP Head：embed_dim → 2048 → 1024，Dropout=0.5
        self.mlp_block = create_mlp_block(
            input_features=embed_dim,
            output_features=mlp_head_units,   # [2048, 1024]
            activation_function=torch.nn.GELU,
            dropout_rate=0.5,
        )

        # 最終線性層：將 MLP 輸出映射到 num_classes（100 類）的 logits
        self.logits_layer = torch.nn.Linear(mlp_head_units[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.create_patch_layer(x)         # (B, N, C*P²)：切割 patch
        x = self.patch_embedding_layer(x)      # (B, N+1, embed_dim)：加入 class token 與位置嵌入
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)           # 逐層 Transformer Encoding，形狀不變
        # HW 修改：對 patch token（不含第 0 個 class token）做平均池化，取得全域表示
        # x[:, 0] 是 class token 的輸出；x[:, 1:] 是 144 個 patch token 的輸出
        x = torch.mean(x[:, 1:], dim=1)        # (B, embed_dim)：在 patch 序列維度取平均
        x = self.mlp_block(x)                  # (B, 1024)：分類 MLP
        x = self.logits_layer(x)               # (B, 100)：輸出各類別的 logits
        return x


# ── 評估工具函式 ───────────────────────────────────────────────────────────────

def calculate_accuracy(
    outputs: torch.Tensor, ground_truth: torch.Tensor
) -> tuple:
    """計算 Top-1 準確度，回傳 (正確數, 總樣本數)。"""
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)  # 將 logits 轉成機率分布
    predictions = torch.argmax(softmaxed_output, dim=1)              # 取機率最高的類別索引
    num_correct = int(torch.sum(torch.eq(predictions, ground_truth)).item())  # 統計預測正確數
    return num_correct, ground_truth.size()[0]


def calculate_accuracy_top_5(
    outputs: torch.Tensor, ground_truth: torch.Tensor
) -> tuple:
    """計算 Top-5 準確度，回傳 (正確數, 總樣本數)。
    注意：此處刻意保留 notebook 原始寫法 predictions[idx, :4]（取前 4 項而非前 5 項），
    以確保輸出與 notebook 完全一致；這是一個 off-by-one 的實作，但為保留原始行為而不修正。
    """
    num_correct = 0
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)   # logits → 機率
    predictions = torch.argsort(softmaxed_output, dim=1, descending=True)  # 依機率由大到小排序索引
    for idx, x in enumerate(ground_truth):
        # 刻意保留 notebook 原始 slice [:4]（僅檢查前 4 個預測，非前 5 個）
        if torch.isin(x, predictions[idx, :4]):
            num_correct += 1
    return num_correct, ground_truth.size(0)


# ── 訓練流程 ───────────────────────────────────────────────────────────────────

def train_network(
    model: torch.nn.Module,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable,
    trainloader: torch.utils.data.DataLoader,
    validloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """執行完整訓練迴圈並於每個 epoch 印出 loss 與 accuracy。"""
    print("Training Started")
    for epoch in range(1, num_epochs + 1):
        sys.stdout.flush()  # 強制即時輸出 log，避免在 pipe 環境下延遲顯示
        train_loss = []
        valid_loss = []
        num_examples_train = 0
        num_correct_train = 0
        num_examples_valid = 0
        num_correct_valid = 0
        num_correct_train_5 = 0
        num_correct_valid_5 = 0

        model.train()  # 切換訓練模式（啟用 Dropout、BatchNorm 更新等）
        for batch in trainloader:
            optimizer.zero_grad()            # 清除上一步的梯度
            x = batch[0].to(device)          # 影像張量移至 GPU/CPU
            y = batch[1].to(device)          # 標籤移至 GPU/CPU
            outputs = model(x)               # 前向傳播，得到 logits (B, 100)
            loss = loss_function(outputs, y) # 計算 Cross-Entropy 損失
            loss.backward()                  # 反向傳播：計算各參數梯度
            optimizer.step()                 # 依梯度更新參數
            train_loss.append(loss.item())

            num_corr, num_ex = calculate_accuracy(outputs, y)      # Top-1 準確度統計
            num_corr_5, _ = calculate_accuracy_top_5(outputs, y)   # Top-5 準確度統計
            num_examples_train += num_ex
            num_correct_train += num_corr
            num_correct_train_5 += num_corr_5

        model.eval()  # 切換評估模式（停用 Dropout）
        with torch.no_grad():  # 停用梯度計算，節省記憶體
            for batch in validloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss.append(loss.item())
                num_corr, num_ex = calculate_accuracy(outputs, labels)
                num_corr_5, _ = calculate_accuracy_top_5(outputs, labels)
                num_examples_valid += num_ex
                num_correct_valid += num_corr
                num_correct_valid_5 += num_corr_5

        # 印出當前 epoch 的訓練與驗證統計
        print(
            f"Epoch: {epoch}, "
            f"Training Loss: {np.mean(train_loss):.4f}, "
            f"Validation Loss: {np.mean(valid_loss):.4f}, "
            f"Training Accuracy: {num_correct_train/num_examples_train:.4f}, "
            f"Validation Accuracy: {num_correct_valid/num_examples_valid:.4f}, "
            f"Training Accuracy Top-5: {num_correct_train_5/num_examples_train:.4f}, "
            f"Validation Accuracy Top-5: {num_correct_valid_5/num_examples_valid:.4f}"
        )


def test_network(
    model: torch.nn.Module,
    loss_function: Callable,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """在測試集上評估模型，印出 loss 與 accuracy。"""
    test_loss = []
    num_examples = 0
    num_correct = 0
    num_correct_5 = 0
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            output = model(images)
            loss = loss_function(output, labels)
            test_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(output, labels)
            num_corr_5, _ = calculate_accuracy_top_5(output, labels)
            num_examples += num_ex
            num_correct += num_corr
            num_correct_5 += num_corr_5
        print(
            f"Test Loss: {np.mean(test_loss):.4f}, "
            f"Test Accuracy: {num_correct/num_examples:.4f}, "
            f"Test Accuracy Top-5: {num_correct_5/num_examples:.4f}"
        )


# ── 主執行區：實例化模型、優化器、損失函數，啟動訓練 ─────────────────────────
model = ViTClassifierModel(
    num_transformer_layers=TRANSFORMER_LAYERS,        # 8 層 Transformer
    embed_dim=PROJECTION_DIM,                         # 64（嵌入維度）
    feed_forward_dim=PROJECTION_DIM * 2,              # 128（FFN 中間層維度）
    num_heads=NUM_HEADS,                              # 4 頭注意力
    patch_size=PATCH_SIZE,                            # 6×6 patch
    num_patches=NUM_PATCHES,                          # 144 個 patch
    mlp_head_units=MLP_HEAD_UNITS,                    # [2048, 1024]
    num_classes=100,                                  # CIFAR100 有 100 類
    batch_size=BATCH_SIZE,                            # 32（class token 形狀依賴此值）
    device=DEVICE,
).to(DEVICE)

# AdamW 優化器：僅更新 requires_grad=True 的參數（排除凍結參數）
optimizer = torch.optim.AdamW(
    params=filter(lambda param: param.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

loss_function = torch.nn.CrossEntropyLoss()  # 多類別分類標準損失函數

# 啟動訓練
train_network(
    model=model,
    num_epochs=NUM_EPOCHS,
    optimizer=optimizer,
    loss_function=loss_function,
    trainloader=trainloader,
    validloader=validloader,
    device=DEVICE,
)

# 測試集評估（對應 notebook 中被 comment 的 cell，保留但預設不執行）
# test_network(
#     model=model,
#     loss_function=loss_function,
#     testloader=testloader,
#     device=DEVICE,
# )