# !/usr/bin/env python
# -*- coding: utf-8 -*-

# plot ref: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import sys
# print(sys.path)
from sklearn import metrics
from GetModel import initialize_model
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from cf_matrix import make_confusion_matrix
from Score import get_all_category_scocre

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def train(model, criterion, optimizer, scheduler, batch_size=4,num_workers=4,num_epochs=25, data_dir=r"",save_model_path=r""):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['train', 'val']} # shuffle = True代表batch_size 中数据也打乱取

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes



    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    best_gt = []
    best_pred = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        preds_list = []
        gt_labels_list = []
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                preds_list = []
                gt_labels_list = []

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
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

                preds_list.extend(preds.cpu().detach().numpy().tolist())
                gt_labels_list.extend(labels.data.cpu().detach().numpy().tolist())
            accuracy,precision_score,recall_score,f1_score = get_all_category_scocre(gt_labels_list,preds_list)
            print(f'{phase} --- accuracy :{accuracy:.4f}，recision_score :{precision_score:.4f}，recall_score :{recall_score:.4f}， f1_score： {f1_score:.4f}')
            plot_confusion_matrix(gt_labels_list,preds_list)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_gt.clear()
                best_pred.clear()
                best_gt = gt_labels_list
                best_pred = preds_list

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_model_path = os.path.join(save_model_path,"model.pth")
    torch.save(model.load_state_dict,save_model_path)
    # return model

def fineTune(nc,pretrained):
    model_conv = torchvision.models.resnet18(pretrained=pretrained)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

def plot_confusion_matrix(gt, pred, save=False):

    cm = metrics.confusion_matrix(gt, pred)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Zero', 'One']
    make_confusion_matrix(cm, group_names=labels, categories=categories, cmap='Blues',
                          title='My Two-class CF Matrix')

    if save is True:
        plt.savefig("./confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("--modelName",type=str,default="resnet18",help="which model is use")
    parser.add_argument("--scheduler",type=float,default=0.1,help="set scheduler to learn")
    parser.add_argument("--batch_size",type=int,default=4,help="once training picture number")
    parser.add_argument("--num_workers", type=int, default=4, help="worker load numbers")
    parser.add_argument("--num_epochs", type=int, default=25, help="training numbers")
    parser.add_argument("--data_dir", type=str, default=r"D:\data\hymenoptera_data", help=r"data dir")
    parser.add_argument("--save_model_path", type=str, default=r"D:\data\hymenoptera_data", help="training dataset path")

    opt = parser.parse_args()
    scheduler = opt.scheduler
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    num_epochs = opt.num_epochs
    data_dir = opt.data_dir
    save_model_path = opt.save_model_path

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = initialize_model("shufflenet_v2_x0_5",num_classes=2,use_pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    train(model, criterion, optimizer_ft, exp_lr_scheduler, batch_size,num_workers,num_epochs,data_dir,save_model_path)