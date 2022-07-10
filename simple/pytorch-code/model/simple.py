# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        
        self.conv = nn.Conv2d(3, 10, 3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        # self.linear = nn.Linear(512,1)
            
    def forward(self, x):
        x = self.conv(x)
        return x

