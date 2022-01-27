from typing import OrderedDict
import torch
from torchvision import models
import torch.nn as nn


class Net_18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=True, embad=50, freeze=False):
        # self.load_state_dict(torch.load('models/9.model'))
        if new_fc:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.01), nn.Linear(512, embad))
        if freeze:
            freeze_names = ['layer4', 'avgpool', 'fc']

            for name, child in self.model.named_children():
                if name not in freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False


class Net_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=1, embad=75, freeze=False):
        if new_fc==1:
            self.model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.LeakyReLU(), nn.Linear(2048, 3000), nn.LeakyReLU(), nn.Linear(
                3000, 3000),nn.Dropout(p=0.5), nn.LeakyReLU(), nn.Linear(3000, 3000), nn.LeakyReLU(), nn.Linear(3000, embad))
        elif new_fc==2:
            self.model.fc = nn.Sequential(nn.Dropout(p=0.1), nn.LeakyReLU(),nn.Linear(2048, embad))
        if freeze:
            un_freeze_names = ['layer4', 'avgpool', 'fc']
            un_freeze_names = ['fc']
            for name, child in self.model.named_children():
                if name not in un_freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False
    def unfreeze(self,un_freeze_names):
        for name, child in self.model.named_children():
            if name not in un_freeze_names:
                for param in child.parameters():
                    param.requires_grad = False



class Net_152(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    def pretrain(self, new_fc=True, embad=75, freeze=False):
        # self.load_state_dict(torch.load('models/9.model'))
        if new_fc:
            self.model.fc = nn.Sequential(nn.Dropout(p=0.1), nn.LeakyReLU(), nn.Linear(2048, 3000), nn.LeakyReLU(), nn.Linear(
                3000, 3000), nn.Dropout(p=0.1), nn.LeakyReLU(), nn.Linear(3000, 3000), nn.LeakyReLU(), nn.Linear(3000, embad))
        if freeze:
            freeze_names = ['layer4', 'avgpool', 'fc']

            for name, child in self.model.named_children():
                if name not in freeze_names:
                    for param in child.parameters():
                        param.requires_grad = False
