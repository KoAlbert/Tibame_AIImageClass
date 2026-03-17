from google.colab import drive
drive.mount('/content/drive')

# if you mount Google drive correctly, the following commands should be able to executed correctly
import os
os.system('ls /content/drive/')
os.chdir("/content/drive/My Drive")
os.chdir("CamVid")

os.system('ls')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision
from torchvision import models
from torchvision.models.vgg import VGG
import random

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from os import path

from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


root_dir   = "/content/drive/My Drive/CamVid/"
train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")

print("training csv exits:{}.format(path.exists(train_file)))
print("validation csv exits:{}.format(path.exists(val_file)))

# the folder to save results for comparison
folder_to_save_validation_result = '/content/drive/My Drive/CamVid/result_comparision/'

if os.path.isdir(folder_to_save_validation_result) == False:
    os.mkdir(folder_to_save_validation_result)


# the number of segmentation classes
num_class =  11 # 32 for original CamVid

h, w      = 256, 256
train_h = 256
train_w = 256
val_h = 256
val_w = 256

## parameters for Solver-Adam in this example
batch_size = 6 #
epochs     = 20 # don't try to improve the performance by simply increasing the training epochs or iterations
lr         = 1e-4    # achieved best results
step_size  = 100 # Won't work when epochs <=100
gamma      = 0.5 #
#

## index for validation images
global_index = 0

# pixel accuracy and mIOU list
pixel_acc_list = []
mIOU_list = []

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

class CamVidDataset(Dataset):

    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5):
        self.data      = pd.read_csv(csv_file, header=None) # ,header=None would read the 1st row of the entire CSV file (normally, 1st row is the title of each column which shouldn't be included.)
        #self.means     = means
        self.n_class   = n_class
        self.flip_rate = flip_rate

        self.resize_h = h
        self.resize_w = w

        if phase == 'train':
            self.new_h = train_h
            self.new_w = train_w
            self.crop = crop
        elif phase == 'val':
            self.flip_rate = 0.
            self.crop = False # False
            self.new_h = val_h
            self.new_w = val_w


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name   = self.data.iloc[idx, 0]
        img_name = root_dir  + img_name
        img = Image.open(img_name).convert('RGB')

        label_name = self.data.iloc[idx, 1]
        label_name = root_dir  + label_name
        label_image = Image.open(label_name)
        label = np.asarray(label_image)

        # In training mode, the crop strategy is random-shift crop.
        # In validation model, it is center crop.
        if self.crop:
            w, h = img.size
            A_x_offset = np.int32(np.random.randint(0, w - self.new_w + 1, 1))[0]
            A_y_offset = np.int32(np.random.randint(0, h - self.new_h + 1, 1))[0]

            img = img.crop((A_x_offset, A_y_offset, A_x_offset + self.new_w, A_y_offset + self.new_h)) # left, top, right, bottom
            label_image = label_image.crop((A_x_offset, A_y_offset, A_x_offset + self.new_w, A_y_offset + self.new_h)) # left, top, right, bottom
        else:
            w, h = img.size
            A_x_offset = int((w - self.new_w)/2)
            A_y_offset = int((h - self.new_h)/2)

            img = img.crop((A_x_offset, A_y_offset, A_x_offset + self.new_w, A_y_offset + self.new_h)) # left, top, right, bottom
            label_image = label_image.crop((A_x_offset, A_y_offset, A_x_offset + self.new_w, A_y_offset + self.new_h)) # left, top, right, bottom

            label_image_h, label_image_w = label_image.size

        # we could try to revise the values in label for reducing the number of segmentation classes
        label = np.array(label_image)

        #label[label==11] = 255 # when using cross-entropy, 255 is a common choice for dont-care class


        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)

        transform = transforms.Compose([ transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        img = transform(img.copy())

        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        # sample = {'X': img, 'Y': target, 'l': label}
        # label = torch.from_numpy(label.copy()).long()
        sample = {'X': img, 'Y': target, 'l': label}  # Return label as (H, W), not one-hot

        return sample


train_data = CamVidDataset(csv_file=train_file, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

# Imports (add torchvision.models for ViT)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import utils, models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from PIL import Image
import pandas as pd

# [Previous unchanged code: dataset, hyperparameters, CamVidDataset, etc.]

# SETR Model with Pre-trained ViT
class SETR(nn.Module):
    def __init__(self, n_class, patch_size=16, embed_dim=768, num_heads=12, num_layers=12, dropout_rate=0.1, pretrained=True):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_class = n_class

        # Load pre-trained ViT-B/16 from torchvision
        if pretrained:
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.patch_embed = vit.conv_proj  # (3, 768, 16, 16) patch projection
            self.pos_embed = vit.encoder.pos_embedding[:, 1:, :]  # Exclude cls token, (1, 196, 768)
            self.transformer = vit.encoder.layers  # Transformer layers
            self.ln = vit.encoder.ln  # Final LayerNorm
        else:
            # Original scratch initialization (for reference)
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = (h // patch_size) * (w // patch_size)  # 256/16 = 16 -> 256 patches
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=dropout_rate, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.ln = nn.LayerNorm(embed_dim)

        # Decoder (naive one)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, n_class, kernel_size=16, stride=16)
        )

        # SETR-PUP
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(embed_dim, 256, kernel_size=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, n_class, kernel_size=4, stride=4, padding=0)
        # )
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, 16, 16)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim), e.g., (B, 256, 768)

        # Positional embedding adjustment
        num_patches_input = (h // self.patch_size) * (w // self.patch_size)  # 256 for 256x256
        if self.pos_embed.shape[1] != num_patches_input:
            # Resize pos_embed from 196 (224x224 ViT default) to 256 (256x256 CamVid)
            pos_embed = self.pos_embed.view(1, 14, 14, self.embed_dim)  # ViT default: 224/16 = 14
            pos_embed = nn.functional.interpolate(pos_embed.permute(0, 3, 1, 2), size=(16, 16), mode='bilinear')
            self.pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, 256, self.embed_dim)

        # Move pos_embed to the same device as x:
        self.pos_embed = self.pos_embed.to(x.device)

        x = x + self.pos_embed  # Add positional encoding

        # Transformer Encoder
        if hasattr(self.transformer, 'layers'):  # Pre-trained ViT case
            for layer in self.transformer:
                x = layer(x)  # Apply each layer sequentially
            x = self.ln(x)  # Final LayerNorm
        else:  # Scratch case
            x = self.transformer(x)
            x = self.ln(x)

        # Reshape to spatial dimensions
        x = x.transpose(1, 2).view(-1, self.embed_dim, h // self.patch_size, w // self.patch_size)  # (B, 768, 16, 16)
        x = self.decoder(x)
        return x

# Initialize SETR with Pre-trained ViT
setr_model = SETR(
    n_class=num_class,
    patch_size=16,
    embed_dim=768,  # ViT-B/16 uses 768
    num_heads=12,  # ViT-B/16 default
    num_layers=12,  # ViT-B/16 default
    dropout_rate=0.1,
    pretrained=True  # Load pre-trained weights
)
if use_gpu:
    setr_model = setr_model.cuda()
    setr_model = nn.DataParallel(setr_model, device_ids=num_gpu)

ts = time.time()
print("Finish cuda loading and model init, time elapsed:", time.time() - ts)

# Optimizer and Scheduler (adjust lr for fine-tuning)
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss(ignore_index=255)

optimizer = optim.AdamW([
    {'params': setr_model.module.patch_embed.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': setr_model.module.transformer.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': setr_model.module.ln.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
    {'params': setr_model.module.decoder.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}
])

scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def save_result_comparison(input_np, output_np):
    #means     = np.array([103.939, 116.779, 123.68]) / 255.

    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])


    global global_index

    original_im_RGB = np.zeros((256,256,3))
    original_im_RGB[:,:,0] = input_np[0,0,:,:]
    original_im_RGB(:,:,1] = input_np[0,1,:,:]
    original_im_RGB[:,:,2] = input_np[0,2,:,:]

    #original_im_RGB[:,:,0] = original_im_RGB[:,:,0] + means[0]
    #original_im_RGB[:,:,1] = original_im_RGB[:,:,1] + means[1]
    #original_im_RGB[:,:,2] = original_im_RGB[:,:,2] + means[2]

    original_im_RGB[:,:,0] = original_im_RGB[:,:,0]*stds[0] + means[0]
    original_im_RGB[:,:,1] = original_im_RGB[:,:,1]*stds[0] + means[1]
    original_im_RGB[:,:,2] = original_im_RGB[:,:,2]*stds[0] + means[2]

    original_im_RGB[:,:,0] = original_im_RGB[:,:,0]*255.0
    original_im_RGB[:,:,1] = original_im_RGB[:,:,1]*255.0
    original_im_RGB[:,:,2] = original_im_RGB[:,:,2]*255.0

    im_seg_RGB = np.zeros((256,256,3))

    # the following version is designed for 11-class version and could still work if the number of classes is fewer.
    for i in range(256):
        for j in range(256):
            if output_np[i,j] == 0:
                im_seg_RGB[i,j,:] = [128, 128, 128]
            elif output_np[i,j] == 1:
                im_seg_RGB[i,j,:] = [128, 0, 0]
            elif output_np[i,j] == 2:
                im_seg_RGB[i,j,:] = [192, 192, 128]
            elif output_np[i,j] == 3:
                im_seg_RGB[i,j,:] = [128, 64, 128]
            elif output_np[i,j] == 4:
                im_seg_RGB[i,j,:] = [0, 0, 192]
            elif output_np[i,j] == 5:
                im_seg_RGB[i,j,:] = [128, 128, 0]
            elif output_np[i,j] == 6:
                im_seg_RGB[i,j,:] = [192, 128, 128]
            elif output_np[i,j] == 7:
                im_seg_RGB[i,j,:] = [64, 64, 128]
            elif output_np[i,j] == 8:
                im_seg_RGB[i,j,:] = [64, 0, 128]
            elif output_np[i,j] == 9:
                im_seg_RGB[i,j,:] = [64, 64, 0]
            elif output_np[i,j] == 10:
                im_seg_RGB[i,j,:] = [0, 128, 192]

    # horizontally stack original image and its corresponding segmentation results
    hstack_image = np.hstack((original_im_RGB, im_seg_RGB))
    new_im = Image.fromarray(np.uint8(hstack_image))

    file_name = folder_to_save_validation_result + str(global_index) + '.jpg'

    global_index = global_index + 1

    new_im.save(file_name)

def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = setr_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))


        val(epoch)

    highest_pixel_acc = max(pixel_acc_list)
    highest_mIOU = max(mIOU_list)

    highest_pixel_acc_epoch = pixel_acc_list.index(highest_pixel_acc)
    highest_mIOU_epoch = mIOU_list.index(highest_mIOU)

    print("The highest mIOU is {} and is achieved at epoch-{}".format(highest_mIOU, highest_mIOU_epoch))
    print("The highest pixel accuracy  is {} and is achieved at epoch-{}".format(highest_pixel_acc, highest_pixel_acc_epoch))

def val(epoch):
    setr_model.eval()
    total_ious = []
    pixel_accs = []

    for iter, batch in enumerate(val_loader):  ## batch is 1 in this case
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
            # Move labels to the same device as inputs (CUDA)
            #labels = Variable(batch['Y'].cuda())
        else:
            inputs = Variable(batch['X'])
            #labels = Variable(batch['Y'])

        output = setr_model(inputs)


        # only save the 1st image for comparison
        if iter == 0:
            print('---------iter={}'.format(iter))
            # generate images
            images = output.data.max(1)[1].cpu().numpy()[:,:,:]
            image = images[0,:,:]
            save_result_comparison(batch['X'], image)

        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, num_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        #target = batch['Y'].cpu().numpy()  # (B, H, W)

        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))

    global pixel_acc_list
    global mIOU_list

    pixel_acc_list.append(pixel_accs)
    mIOU_list.append(np.nanmean(ious))

def iou(pred, target):
    ious = []
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


## perform training and validation
val(0)  # show the accuracy before training
train()