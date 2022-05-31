import torch
import torch.nn as nn

from layer import FC, IncidenceConvolution


class CNFNet(nn.Module):
    def __init__(self, opt):
        super(CNFNet, self).__init__()
        self.opt = opt

        # initialize CNFNet
        self.energy_kernel = IncidenceConvolution(opt)
        self.fc = FC(opt)

    def forward(self, inc, f):
        # calculate energy kernel
        y = self.energy_kernel(inc)

        # concatenate with CNF properties
        z = torch.cat((f, y), 1)

        # connect to fully-connected layers
        z = self.fc(z)

        return z
