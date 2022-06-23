
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




class SiameseNet(nn.Module):
    def __init__(self, base_model, dropout_rate=0.0, bayesian=False):
        super(SiameseNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # Encoding Blocks
        self.init_block = nn.Sequential(*list(base_model.children())[:-1])

        # Regressor
        self.fc_dim_reduce = nn.Linear(2048, 1024)

        self.FinalLayer_mat = nn.Sequential(
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
            nn.Linear(256, 12)
        )


    def forward_one(self, x):
        # Conv
        x_conv = self.init_block(x)
        #print("shape of output feature : {}".format(x_conv.shape))
        x_linear = x_conv.view(x_conv.size(0), -1)
        x_linear = self.fc_dim_reduce(x_linear)
        #x_linear = F.relu(x_linear)
        #x_linear = F.tanh(x_linear) # changed Relu to Tanh
        #x_linear = F.tanhshrink(x_linear)

        #x_linear = F.leaky_relu(x_linear, negative_slope=0.2, inplace=False)
        #print("shape of x_linear : {}".format(x_linear.shape))#



        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_linear = F.dropout(x_linear, p=self.dropout_rate, training=dropout_on)
            #x_linear = torch.mul(x_linear, self.dropout_rate)
            #print("shape of x_linear : {}".format(x_linear.shape))

        #trans = self.fc_trans(x_linear)
        #rot = self.fc_rot(x_linear)

        #return trans, rot
        return x_linear, x_conv
        #return x_linear

    def forward(self, x1, x2):
        x1_, x_conv1 = self.forward_one(x1)
        #x1_ = self.forward_one(x1)
        x2_, x_conv2 = self.forward_one(x2)
        #x2_ = self.forward_one(x2)
        diff = torch.abs(x1_ - x2_)
        out = torch.cat((x1_, x2_, diff), 1)
        mat = self.FinalLayer_mat(out)
        return mat, x_conv1, x_conv2
        #return rot, trans


class SiameseNetClassify(nn.Module):
    def __init__(self, base_model, dropout_rate=0.0, bayesian=False):
        super(SiameseNetClassify, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # Encoding Blocks
        self.init_block = nn.Sequential(*list(base_model.children())[:-1])

        # Regressor
        #self.fc_dim_reduce = nn.Linear(2048, 1024)

        self.FinalLayer_mat = nn.Sequential(
            # nn.Linear(4096 * 3, 1024),
            nn.Linear(2048, 3000),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            #nn.PReLU(),
            nn.Linear(3000, 3000),
            nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            #nn.PReLU(),
            nn.Linear(3000, 2353)
        )


    def forward_one(self, x):
        # Conv
        x_conv = self.init_block(x)
        #print("shape of output feature : {}".format(x_conv.shape))
        x_linear = x_conv.view(x_conv.size(0), -1)
        #x_linear = self.fc_dim_reduce(x_linear)
        #x_linear = F.relu(x_linear)
        #x_linear = F.tanh(x_linear) # changed Relu to Tanh
        #x_linear = F.tanhshrink(x_linear)

        #x_linear = F.leaky_relu(x_linear, negative_slope=0.2, inplace=False)
        #print("shape of x_linear : {}".format(x_linear.shape))#



        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_linear = F.dropout(x_linear, p=self.dropout_rate, training=dropout_on)
            #x_linear = torch.mul(x_linear, self.dropout_rate)
            #print("shape of x_linear : {}".format(x_linear.shape))

        #trans = self.fc_trans(x_linear)
        #rot = self.fc_rot(x_linear)

        #return trans, rot
        return x_linear, x_conv
        #return x_linear

    def forward(self, x1, x2):
        x1_, x_conv1 = self.forward_one(x1)
        #x1_ = self.forward_one(x1)
        x2_, x_conv2 = self.forward_one(x2)
        #x2_ = self.forward_one(x2)
        diff = torch.abs(x1_ - x2_)
        #out = torch.cat((x1_, x2_, diff), 1)
        out = self.FinalLayer_mat(diff)
        return out, x_conv1, x_conv2
        #return rot, trans