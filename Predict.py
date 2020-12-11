# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
import argparse
from GetModel import initialize_model
import cv2
import numpy as np


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def visualize_model(model,batch_size=4,num_workers=4,data_dir=r""):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model= model.to(device) # this is very important
    img = cv2.imread(data_dir)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to cxhxw
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img)
    input = img.to(device)

    if input.ndimension() == 3:
        input = input.unsqueeze(0)
        outputs = model(input)
        _, preds = torch.max(outputs, 1)
        print(f"predict: {preds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("--batch_size", type=int, default=4, help="once training picture number")
    parser.add_argument("--num_workers", type=int, default=4, help="worker load numbers")
    parser.add_argument("--data_dir", type=str, default=r"D:\data\hymenoptera_data", help=r"data dir")

    model = initialize_model("shufflenet_v2_x0_5",2,use_pretrained=False)
    model.load_state_dict(torch.load("D:\data\hymenoptera_data\model.pth"))
    model.eval()
    visualize_model(model,batch_size=4,num_workers=4,data_dir=r"D:\data\hymenoptera_data\test\10870992_eebeeb3a12.jpg")
