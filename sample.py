# File Management
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

# Utilities
from __future__ import print_function
import argparse, random, copy
import numpy as np
import tqdm

# Torch imports
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

import torchxrayvision as xrv

# SKLearn
import sklearn
import scikitplot as skplt
from sklearn.model_selection import train_test_split

# Premade Transformers
from transformers import AutoModel

# Configurations for Siamese Network
config = {
    "d_model": 512,
    "dropout": 0.2,
    "num_classes": 2,
    "lr": 1e-3
}

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        """
        The siamese network architecture makes use of two parallel neural
        networks that learn to, in this case, be able to successfully
        encode and distinguish between two images.
        """
        super(SiameseNetwork, self).__init__()

        # 101-elastic is trained on PadChest, NIH, CheXpert, and MIMIC datasets
        self.cnn1 = xrv.autoencoders.ResNetAE(weights="101-elastic")
        self.nclasses = config["num_classes"]

        outdim = 512 * 3 * 3 * 2

        for param in self.cnn1.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(outdim, config["d_model"])
        self.dropout = nn.Dropout(config["dropout"], inplace=False)
        self.fc_final = nn.Linear(config["d_model"], config["num_classes"])
        self.learning_rate = config["lr"]

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output["z"].view(-1, 512*3*3)
        return output

    def forward(self, input1, input2):
        # Parallel networks in action
        prev_image_features = self.forward_once(input1)
        curr_image_features = self.forward_once(input2)

        # Combine the features, and generate the output
        image_features = torch.cat((prev_image_features, curr_image_features), 1)
        image_features = F.relu(self.fc(image_features))
        image_features = self.dropout(image_features)
        classifier_output = self.fc_final(image_features)

        return classifier_output

class SiameseDataset(Dataset):
    def __init__(self, data, prev_pkl, current_pkl):
        """
        The SiameseDataset object stores the data into something neat and accessible.
        Also, makes the images have 3->1 channel (grayscale).
        """
        super(SiameseDataset, self).__init__()
        self.data = data
        self.prev_pkl = prev_pkl
        self.current_pkl = current_pkl
        self.greyscale = torchvision.transforms.Grayscale(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_1 = self.prev_pkl[self.data["object_id"][index]]
        image_1 = self.greyscale(image_1)
        image_2 = self.current_pkl[self.data["subject_id"][index]]
        image_2 = self.greyscale(image_2)
        if self.data["comparison"][index] == "no change":
          target = 0
        else:
          target = 1

        return {"prev_img": image_1, "curr_img": image_2, "change": targ

def train(model, device, train_loader, optimizer, epoch):
    """
    This uses CrossEntropyLoss. Although BinaryEntropyLoss function could
    also be used, the documentation preferred using CrossEntropyLoss.

    Other than that, this is essentially training the model.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
        targets = batch["change"].type(torch.LongTensor)
        images_1 = batch["prev_img"].to(device)
        images_2 = batch["curr_img"].to(device)
        targets = batch["change"].to(device)

        optimizer.zero_grad()
        outputs = model(images_1, images_2)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(images_1), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """
    This function tells the accuracy and loss of the model.
    """
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            targets = batch["change"].type(torch.LongTensor)
            images_1 = batch["prev_img"].to(device)
            images_2 = batch["curr_img"].to(device)
            targets = batch["change"].to(device)

            outputs = model(images_1, images_2)
            test_loss += criterion(outputs, targets).sum().item()
            pred = outputs.argmax(1)
            correct += pred.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

"""
------------- RUN THE TRAIN-TEST LOOP -------------
"""

# Load dataset pt. 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df, test_df = train_test_split(csv_df, shuffle=True)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_ds = SiameseDataset(train_df, prevpkl, currentpkl)
test_ds = SiameseDataset(test_df, prevpkl, currentpkl)
train_data_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_data_loader = DataLoader(test_ds, batch_size=4, shuffle=True)

# Train-test Loop
model = SiameseNetwork().to(device)
optimizer = optim.SGD(model.parameters(), lr=config["lr"])

epochs = 100
for epoch in range(1, epochs + 1):
    train(model, device, train_data_loader, optimizer, epoch)
    test(model, device, test_data_loader)
