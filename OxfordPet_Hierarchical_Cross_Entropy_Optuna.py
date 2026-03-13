!pip install --upgrade gdown

# installing optuna
!pip install optuna

from google.colab import drive
import os

In_this_space_flag = True

if In_this_space_flag == True and os.path.isdir("/content/oxford-iiit-pet/") == False:
  # 1 do everything in this temporary space
  os.system("gdown https://drive.google.com/uc?id=1ymthIy1mSTw0RFhTCd9NCmA-pJi6zjtw")
  os.system("unzip oxford-iiit-pet.zip")
  os.remove("oxford-iiit-pet.zip")
else:
  drive.mount('/content/drive')
  assert os.path.isdir("/content/drive/My Drive/oxford-iiit-pet/") == True

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


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, num_class, file_root, mode, transform=None):
        data_dic = {str(num):[] for num in  range(num_class)}

        images = []
        labels = open(data_list).readlines()
        for line in labels:
            items = line.strip('\n').split()
            img_name = items[0]
            label = str(int(items[1]) - 1)

            if int(label) > 23:
                continue

            data_dic[label].append(img_name)

        for cls in range(num_class):
            file_list = data_dic[str(cls)]
            file_num = len(file_list)
            imgs_list = [(file_root + file_list[i] + '.jpg', cls) for i in range(file_num)]
            images = images + imgs_list

        self.images = images
        self.transform = transform


    def __getitem__(self, index):
        img_name, label = self.images[index]
        assert os.path.exists(img_name) == True
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return (img, label) if label is not None else img

    def __len__(self):
        return len(self.images)

def OxfordPet_dataloader(file_root, data_batch, train_data_list, val_data_list, num_class, img_size):

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(img_size),
        #transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.CenterCrop(img_size),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomRotation(degrees=5),
        #transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    }

    train_set = OxfordPetDataset(train_data_list, num_class, file_root, 'train', data_transforms['train'])
    val_set = OxfordPetDataset(val_data_list, num_class,file_root, 'val',data_transforms['val'])


    image_datasets = {'train': train_set , 'val': val_set }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=data_batch, shuffle=True, num_workers=8) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return {'dataloaders':dataloaders, 'dataset_sizes':dataset_sizes}

def HierarchyLoss(outputs, labels, w):

    input_p = torch.softmax(outputs,dim=1)

    # changes from many classes down to 2 classes for both input and target
    # Sum the probabilities for all the cat breeds to get probability it's a cat.  Same for dog
    cats = torch.sum(input_p[:,0:12],dim=1).view(input_p.shape[0],1)
    dogs = torch.sum(input_p[:,12:37],dim=1).view(input_p.shape[0],1)

    # format new inputs and new targets for 2 classes
    new_input = torch.cat([cats,dogs],-1)
    new_target = labels > 11
    new_target = new_target.long()

    # Finish calculation for cross-entropy using new inputs and targets
    Species_loss = torch.nn.functional.nll_loss(torch.log(new_input), new_target)

    input_p = torch.softmax(outputs,dim=-1)

    breed_loss = torch.nn.functional.nll_loss(torch.log(input_p), labels)

    #return w*ce_species+(1-w)*ce_breed

    #return  w*Species_loss + (1-w)*breed_loss
    return  w*Species_loss + breed_loss

def train_model_HierarchicalLoss(model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, num_class, weighting, save_model_flag=False ):

    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print('start training!')
                since_training = time.time()
            else:
                model.eval()   # Set model to evaluate mode
                print('start evaluation!')
                since_val = time.time()

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    #loss = criterion(outputs, labels)
                    loss  = HierarchyLoss(outputs, labels, weighting)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                final_train_acc = epoch_acc

            # deep copy the model
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


    return {'final epoch train accuracy': final_train_acc.cpu().detach().numpy(), 'final epoch val accuracy': final_val_acc.cpu().detach().numpy(),\
     'highest val accuracy': best_acc.cpu().detach().numpy(), 'highest val accuracy epoch': epoch_on_best_acc, 'training time in mins': time_elapsed_mins}

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

# check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('torch.cuda.is_available()={}'.format(torch.cuda.is_available()))


#
if In_this_space_flag == True:
  ## version 1: download file
  # data folder
  VOC_root_dir = "/content/YOLO_data_model/"
  train_data_list = '/content/oxford-iiit-pet/file_train_list.txt'
  val_data_list = '/content/oxford-iiit-pet/file_val_list.txt'
  file_root = '/content/oxford-iiit-pet/images/'
else:
  ## version 2: access the folder in your google drive
  train_data_list = '/content/drive/My Drive/oxford-iiit-pet/file_train_list.txt'
  val_data_list = '/content/drive/My Drive/oxford-iiit-pet/file_val_list.txt'
  file_root = '/content/drive/My Drive/oxford-iiit-pet/images/'


num_class = 24#37
img_size = 224

num_epochs = 10
lr = 0.0002
num_batch = 12
#w = 0.11299

weight_list = []
val_accuracy = []


def oxford_pet_hierarchical(trial):

  ## 1. assign a random weighting
  #cfg = { 'weighting' : trial.suggest_uniform('weighting', 0.01, 0.5)}
  cfg = { 'weighting' : trial.suggest_uniform('weighting', 0.01, 1.0)}
  w = cfg['weighting']

  #data preparation
  data_dic = OxfordPet_dataloader(file_root, num_batch, train_data_list, val_data_list, num_class, img_size)

  #define model: the model will be downloaded to a temp folder
  model_ft = models.resnet50(pretrained=True)

  #some surgery for the pretrained model, e.g., alexnet
  model_ft.fc = nn.Sequential(
          nn.Linear(2048, 2048), #256 * 6 * 6
          nn.ReLU(),
          nn.Linear(2048, 2048),
          nn.ReLU(),
          nn.Linear(2048, num_class))

  #print(model_ft)
  model_ft = model_ft.to(device)

  ## define criterion
  #criterion = nn.CrossEntropyLoss()

  ## define solver
  #optimizer = optim.SGD(model_ft.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=0.0001)
  optimizer = optim.Adam(model_ft.parameters(), lr=lr)


  ## define learning rate decay policy: step size & gamma
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

  ## training the defined model for the designated epochs
  result_dic = train_model_HierarchicalLoss(model_ft, optimizer, exp_lr_scheduler, num_epochs=num_epochs, dataloaders = data_dic['dataloaders']\
    , dataset_sizes = data_dic['dataset_sizes'], device = device, num_class=num_class, weighting = w)

  highest_val_accuracy = result_dic['highest val accuracy'].round(4)

  print('In {} epochs, weights={}, highest val accuracy={} is achieved at epoch-{}'.format(num_epochs, w, highest_val_accuracy, result_dic['highest val accuracy epoch'] ) ) 

  weight_list.append(w)
  val_accuracy.append(highest_val_accuracy)

  return highest_val_accuracy

if __name__ == '__main__':

  sampler = optuna.samplers.TPESampler()

  study = optuna.create_study(sampler=sampler, direction='maximize')
  study.optimize(func=oxford_pet_hierarchical, n_trials=200)

  #print(weight_list)
  #print(val_accuracy)

  max_value = max(val_accuracy)
  max_index = val_accuracy.index(max_value)

  print('--max accuracy={} with w={}'.format(max_value, weight_list[max_index]))
