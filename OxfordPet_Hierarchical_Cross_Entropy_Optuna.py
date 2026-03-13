# ============================================================================
# 程式名稱：OxfordPet_Hierarchical_Cross_Entropy_Optuna.py
# 程式目的：使用「階層式交叉熵損失 (Hierarchical Cross Entropy)」搭配 Optuna
#          超參數搜索，訓練 ResNet-50 模型對 Oxford-IIIT Pet 資料集進行寵物品種分類。
#
# 核心概念：
#   1. 遷移學習 (Transfer Learning)：使用 ImageNet 預訓練的 ResNet-50
#   2. 階層式損失 (Hierarchical Loss)：同時優化「貓/狗」二分類與「24品種」細分類
#   3. Optuna 超參數最佳化：自動搜索階層損失中的最佳權重 w
#
# 總損失 = w * Species_loss + 1 * Breed_loss
# ============================================================================

# === 第一部分：環境安裝 ===
# 安裝 gdown（用於從 Google Drive 下載大型檔案）
!pip install --upgrade gdown

# 安裝 optuna（超參數自動最佳化框架）
!pip install optuna

# === 第二部分：資料準備 ===
from google.colab import drive
import os

# 旗標設定：決定資料來源
# True  = 在 Colab 臨時空間下載資料（不掛載 Google Drive）
# False = 掛載 Google Drive，從雲端硬碟讀取資料
In_this_space_flag = True

if In_this_space_flag == True and os.path.isdir("/content/oxford-iiit-pet/") == False:
  # 方式一：從 Google Drive 下載壓縮檔，解壓後刪除 zip 節省空間
  os.system("gdown https://drive.google.com/uc?id=1ymthIy1mSTw0RFhTCd9NCmA-pJi6zjtw")
  os.system("unzip oxford-iiit-pet.zip")
  os.remove("oxford-iiit-pet.zip")
else:
  # 方式二：掛載 Google Drive，直接存取雲端硬碟中的資料
  drive.mount('/content/drive')
  assert os.path.isdir("/content/drive/My Drive/oxford-iiit-pet/") == True

# === 第三部分：匯入所需套件 ===
from __future__ import print_function, division
from xml.dom import HierarchyRequestErr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import shutil
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import random


# ============================================================================
# 第四部分：自定義資料集類別 (Dataset)
# 繼承 torch.utils.data.Dataset，自訂資料讀取邏輯
# 功能：讀取文字標註檔，建立 (圖片路徑, 類別標籤) 的對應清單
# ============================================================================
class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, num_class, file_root, mode, transform=None):
        """
        參數說明：
            data_list  : 標註檔路徑，每行格式為 "圖片名稱 類別編號"
            num_class  : 類別總數（本實驗使用前 24 類）
            file_root  : 圖片資料夾的根目錄路徑
            mode       : 'train' 或 'val'（目前未在內部使用，保留給未來擴充）
            transform  : 資料預處理與增強的 transforms 組合
        """
        # 建立字典，key = 類別編號(字串"0"~"23")，value = 該類別的圖片名稱列表
        data_dic = {str(num):[] for num in  range(num_class)}

        images = []
        # 逐行讀取標註檔
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]             # 圖片檔名（不含副檔名）
            label = str(int(items[1]) - 1)  # 原始標籤從 1 開始，轉換為從 0 開始

            # 只保留前 24 個類別（類別 0~23），跳過類別 24 以上的資料
            if int(label) > 23:
                continue

            # 將圖片名稱加入對應類別的列表中
            data_dic[label].append(img_name)

        # 將字典轉換為 (圖片完整路徑, 類別編號) 的 tuple 列表
        for cls in range(num_class):
            file_list = data_dic[str(cls)]
            file_num = len(file_list)
            imgs_list = [(file_root + file_list[i] + '.jpg', cls) for i in range(file_num)]
            images = images + imgs_list

        self.images = images        # 儲存所有 (路徑, 標籤) 的列表
        self.transform = transform  # 儲存預處理 transform


    def __getitem__(self, index):
        """根據索引取得一筆資料，回傳 (圖片 tensor, 類別標籤)"""
        img_name, label = self.images[index]
        assert os.path.exists(img_name) == True  # 確認圖片檔案存在
        img = Image.open(img_name).convert('RGB')  # 讀取圖片並轉為 RGB 格式

        if self.transform is not None:
            img = self.transform(img)  # 套用資料預處理與增強
        return (img, label) if label is not None else img

    def __len__(self):
        """回傳資料集的總筆數"""
        return len(self.images)


# ============================================================================
# 第五部分：資料載入器函式 (DataLoader)
# 定義訓練集與驗證集的資料前處理方式，並建立 DataLoader
# ============================================================================
def OxfordPet_dataloader(file_root, data_batch, train_data_list, val_data_list, num_class, img_size):

    data_transforms = {
    # --- 訓練集：加入資料增強減少過擬合 ---
    'train': transforms.Compose([
        transforms.Resize(img_size),             # 將圖片最短邊縮放到 img_size
        #transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.CenterCrop(img_size),         # 從中心裁切出正方形
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomRotation(degrees=5),    # 隨機旋轉 ±5 度
        #transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),       # 50% 機率水平翻轉
        transforms.ToTensor(),                   # PIL Image 轉為 Tensor [0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 標準化
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    # --- 驗證集：不做隨機增強，只做基本處理 ---
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    }

    # 建立訓練集與驗證集的 Dataset 物件
    train_set = OxfordPetDataset(train_data_list, num_class, file_root, 'train', data_transforms['train'])
    val_set = OxfordPetDataset(val_data_list, num_class,file_root, 'val',data_transforms['val'])

    # 組合成字典，方便用 'train'/'val' 索引
    image_datasets = {'train': train_set , 'val': val_set }
    # DataLoader：自動分批、打亂順序、多執行緒載入
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=data_batch, shuffle=True, num_workers=8) for x in ['train', 'val']}
    # 記錄資料集大小，用於計算平均 loss 和 accuracy
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return {'dataloaders':dataloaders, 'dataset_sizes':dataset_sizes}


# ============================================================================
# 第六部分：階層式交叉熵損失函式 (Hierarchical Cross Entropy Loss) — 核心
#
# 原理：將 24 個品種的 softmax 機率「聚合」成 2 個物種(貓/狗)的機率，
#       分別計算物種層級與品種層級的交叉熵損失，用權重 w 平衡兩者。
# ============================================================================
def HierarchyLoss(outputs, labels, w):
    """
    outputs : 模型原始輸出 logits [batch_size, 24]
    labels  : 真實類別標籤 0~23
    w       : 物種損失的權重（Optuna 搜索的超參數）
    回傳    : w * Species_loss + Breed_loss
    """

    # --- 第一層：物種損失 (Species Loss) ---
    input_p = torch.softmax(outputs,dim=1)  # logits 轉為機率分布

    # 聚合品種機率為物種機率：
    # 類別 0~11 (貓品種) 機率加總 = P(貓)
    # 類別 12~23 (狗品種) 機率加總 = P(狗)
    cats = torch.sum(input_p[:,0:12],dim=1).view(input_p.shape[0],1)
    dogs = torch.sum(input_p[:,12:37],dim=1).view(input_p.shape[0],1)

    # 組合為 [batch, 2] 的機率：[P(貓), P(狗)]
    new_input = torch.cat([cats,dogs],-1)
    # 物種標籤：類別 > 11 為狗(1)，否則為貓(0)
    new_target = labels > 11
    new_target = new_target.long()

    # 計算物種層級的交叉熵 (log + nll_loss = cross_entropy)
    Species_loss = torch.nn.functional.nll_loss(torch.log(new_input), new_target)

    # --- 第二層：品種損失 (Breed Loss) ---
    input_p = torch.softmax(outputs,dim=-1)
    # 標準 24 類交叉熵損失
    breed_loss = torch.nn.functional.nll_loss(torch.log(input_p), labels)

    #return w*ce_species+(1-w)*ce_breed

    #return  w*Species_loss + (1-w)*breed_loss
    # w 越大 → 越強調貓狗二分類；w 越小 → 越專注品種細分類
    return  w*Species_loss + breed_loss


# ============================================================================
# 第七部分：訓練函式（使用階層式損失）
# 標準 PyTorch 訓練迴圈，但用 HierarchyLoss 取代一般 CrossEntropyLoss
# ============================================================================
def train_model_HierarchicalLoss(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, num_class, weighting, save_model_flag=False ):

    best_acc = 0.0  # 歷史最佳驗證準確率

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))


        # 每個 epoch 分為訓練和驗證兩階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 訓練模式：啟用 Dropout、BatchNorm 用 batch 統計量
                print('start training!')
                since_training = time.time()
            else:
                model.eval()   # 評估模式：關閉 Dropout、BatchNorm 用全域統計量
                print('start evaluation!')
                since_val = time.time()

            running_loss = 0.0      # 累計總損失
            running_corrects = 0    # 累計正確預測數


            # 逐批次迭代資料
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 圖片搬到 GPU
                labels = labels.to(device)  # 標籤搬到 GPU

                # 清除上一步梯度
                optimizer.zero_grad()

                # 前向傳播（驗證時不計算梯度以節省記憶體）
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)            # 前向傳播得到 logits [batch, 24]
                    _, preds = torch.max(outputs, 1)   # 取最大機率類別為預測

                    # 使用階層式損失
                    #loss = criterion(outputs, labels)
                    loss  = HierarchyLoss(outputs, labels, weighting)

                    # 反向傳播 + 參數更新（僅訓練階段）
                    if phase == 'train':
                        loss.backward()    # 計算梯度
                        optimizer.step()   # 更新參數

                # 累計統計量
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 訓練階段結束後執行學習率衰減
            if phase == 'train':
                scheduler.step()

            # 計算本 epoch 平均損失與準確率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                final_train_acc = epoch_acc

            # 更新最佳驗證準確率
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                epoch_on_best_acc = epoch

            if phase == 'val':
                final_val_acc = epoch_acc
                print('{} Loss: {:.4f} Acc: {:.4f} best_acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_acc))

            if phase == 'train':
                time_elapsed = time.time() - since_training
                time_elapsed_mins = time_elapsed // 60
                #print('Training a epoch for {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            else:
                time_elapsed = time.time() - since_val
                time_elapsed_mins = time_elapsed // 60
                #print('Validating a epoch for {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    # 回傳訓練結果摘要
    return {'final epoch train accuracy': final_train_acc.cpu().detach().numpy(), 'final epoch val accuracy': final_val_acc.cpu().detach().numpy(),\n     'highest val accuracy': best_acc.cpu().detach().numpy(), 'highest val accuracy epoch': epoch_on_best_acc, 'training time in mins': time_elapsed_mins}

# ============================================================================
# 第八部分：Optuna 超參數搜索設定
# ============================================================================
# 重新匯入套件（Colab notebook 環境中確保可用）
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import cv2
import random
from decimal import Decimal
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict


last_highest_val_accuracy = 0

# 檢測 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('torch.cuda.is_available()={}'.format(torch.cuda.is_available()))


# 根據旗標設定資料路徑
if In_this_space_flag == True:
  ## 方式一：Colab 臨時空間
  # data folder
  VOC_root_dir = "/content/YOLO_data_model/"
  train_data_list = '/content/oxford-iiit-pet/file_train_list.txt'   # 訓練集標註檔
  val_data_list = '/content/oxford-iiit-pet/file_val_list.txt'       # 驗證集標註檔
  file_root = '/content/oxford-iiit-pet/images/'                     # 圖片資料夾
else:
  ## 方式二：Google Drive
  train_data_list = '/content/drive/My Drive/oxford-iiit-pet/file_train_list.txt'
  val_data_list = '/content/drive/My Drive/oxford-iiit-pet/file_val_list.txt'
  file_root = '/content/drive/My Drive/oxford-iiit-pet/images/'


# === 訓練超參數 ===
num_class = 24#37      # 前 24 類（12貓+12狗），原始 37 類
img_size = 224         # 圖片尺寸（ResNet 標準輸入）

num_epochs = 10        # 每次試驗訓練 10 個 epoch
lr = 0.0002            # Adam 學習率
num_batch = 12         # batch size
#w = 0.11299           # 已改由 Optuna 搜索

weight_list = []       # 記錄每次試驗的 w 值
val_accuracy = []      # 記錄每次試驗的最高驗證準確率


# ============================================================================
# 第九部分：Optuna 目標函式
# 每次試驗：Optuna 建議 w → 訓練模型 → 回傳最高驗證準確率
# ============================================================================
def oxford_pet_hierarchical(trial):

  ## 1. Optuna 用 TPE 演算法建議 w 值（範圍 0.01~1.0）
  #cfg = { 'weighting' : trial.suggest_uniform('weighting', 0.01, 0.5)}
  cfg = { 'weighting' : trial.suggest_uniform('weighting', 0.01, 1.0)}
  w = cfg['weighting']

  # 2. 建立 DataLoader
  data_dic = OxfordPet_dataloader(file_root, num_batch, train_data_list, val_data_list, num_class, img_size)

  # 3. 載入 ImageNet 預訓練 ResNet-50（遷移學習）
  model_ft = models.resnet50(pretrained=True)

  # 4. 修改全連接層：1000 類 → 24 類
  #    Linear(2048->2048) -> ReLU -> Linear(2048->2048) -> ReLU -> Linear(2048->24)
  model_ft.fc = nn.Sequential(
          nn.Linear(2048, 2048), #256 * 6 * 6
          nn.ReLU(),
          nn.Linear(2048, 2048),
          nn.ReLU(),
          nn.Linear(2048, num_class))

  #print(model_ft)
  model_ft = model_ft.to(device)  # 模型搬到 GPU

  ## define criterion
  #criterion = nn.CrossEntropyLoss()  # 改用 HierarchyLoss

  ## define solver
  #optimizer = optim.SGD(model_ft.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=0.0001)
  # 5. Adam 優化器
  optimizer = optim.Adam(model_ft.parameters(), lr=lr)


  # 6. 學習率衰減：每 5 epoch 乘以 0.1
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

  # 7. 開始訓練
  result_dic = train_model_HierarchicalLoss(model_ft, optimizer, exp_lr_scheduler, num_epochs=num_epochs, dataloaders = data_dic['dataloaders']\
    , dataset_sizes = data_dic['dataset_sizes'], device = device, num_class=num_class, weighting = w)

  # 8. 取得最高驗證準確率
  highest_val_accuracy = result_dic['highest val accuracy'].round(4)

  print('In {} epochs, weights={}, highest val accuracy={} is achieved at epoch-{}'.format(num_epochs, w, highest_val_accuracy, result_dic['highest val accuracy epoch'] ) ) 

  weight_list.append(w)
  val_accuracy.append(highest_val_accuracy)

  # 回傳給 Optuna（目標：最大化此值）
  return highest_val_accuracy


# ============================================================================
# 第十部分：主程式 — 啟動 Optuna 搜索
# ============================================================================
if __name__ == '__main__':

  # TPE 取樣器：貝葉斯優化，根據過去試驗結果智慧選擇下一個 w
  sampler = optuna.samplers.TPESampler()

  # 建立 study：目標「最大化」驗證準確率
  study = optuna.create_study(sampler=sampler, direction='maximize')
  # 執行 200 次試驗，Optuna 自動收斂到最佳 w
  study.optimize(func=oxford_pet_hierarchical, n_trials=200)

  #print(weight_list)
  #print(val_accuracy)

  # 找出最高準確率及對應的 w
  max_value = max(val_accuracy)
  max_index = val_accuracy.index(max_value)

  # 輸出最終結果
  print('--max accuracy={} with w={}'.format(max_value, weight_list[max_index]))