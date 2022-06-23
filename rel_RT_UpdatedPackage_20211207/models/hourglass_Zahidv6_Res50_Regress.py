
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import torchvision
import glob
import re
import warnings
warnings.filterwarnings("ignore")
#lgr = logging.getLogger(__name__)


class RegressNet(nn.Module):
    def __init__(self, dropout_rate=0.0, bayesian=False):
        super(RegressNet, self).__init__()
        #self.conv_block1 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Regressor
        self.fc_dim_reduce = nn.Linear(14 * 14 * 32, 1024)
        #self.fc_trans = nn.Linear(1024, 3)
        #self.fc_rot = nn.Linear(1024, 4)

        self.FinalLayer_rot = nn.Sequential(
            # nn.Linear(4096 * 3, 1024),
            nn.Linear(1024 * 3, 1024),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            nn.PReLU(),
            nn.Linear(1024, 256),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            nn.PReLU(),
            nn.Linear(256, 4)
        )

        self.FinalLayer_tra = nn.Sequential(
            # nn.Linear(4096 * 3, 1024),
            nn.Linear(1024 * 3, 1024),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            nn.PReLU(),
            nn.Linear(1024, 256),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            nn.PReLU(),
            nn.Linear(256, 3)
        )

    def forward_one(self, x_conv):
        #print("shape of output feature : {}".format(x_conv.shape))
        x_conv = self.cnn_layers(x_conv)
        x_linear = x_conv.view(x_conv.size(0), -1)
        x_linear = self.fc_dim_reduce(x_linear)
        #x_linear = F.relu(x_linear)
        #x_linear = F.tanh(x_linear) # changed Relu to Tanh
        #x_linear = F.tanhshrink(x_linear)

        x_linear = F.leaky_relu(x_linear, negative_slope=0.2, inplace=False)
        #print("shape of x_linear : {}".format(x_linear.shape))#

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_linear = F.dropout(x_linear, p=self.dropout_rate, training=dropout_on)
            #x_linear = torch.mul(x_linear, self.dropout_rate)
            #print("shape of x_linear : {}".format(x_linear.shape))'''

        #trans = self.fc_trans(x_linear)
        #rot = self.fc_rot(x_linear)

        #return trans, rot
        return x_linear
        #return x_linear

    def forward(self, x_conv1, x_conv2):
        x1_ = self.forward_one(x_conv1)
        x2_ = self.forward_one(x_conv2)
        diff = torch.abs(x1_ - x2_)
        out = torch.cat((x1_, x2_, diff), 1)
        rot = self.FinalLayer_rot(out)
        trans = self.FinalLayer_tra(out)
        return rot, trans