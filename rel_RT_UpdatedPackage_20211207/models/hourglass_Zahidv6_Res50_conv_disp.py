
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




class HourglassNet(nn.Module):
    def __init__(self, base_model, sum_mode=False, dropout_rate=0.0, bayesian=False):
        super(HourglassNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        self.sum_mode = sum_mode

        # Encoding Blocks
        self.init_block = nn.Sequential(*list(base_model.children())[:4])

        self.res_block1 = base_model.layer1
        self.res_block2 = base_model.layer2
        self.res_block3 = base_model.layer3
        self.res_block4 = base_model.layer4

        # Decoding Blocks
        if sum_mode:
            self.deconv_block1 = nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            #self.deconv_block4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            #self.deconv_block5 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block1 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.conv_block2 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.conv_block3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

            print("This")
        else:
            self.deconv_block1 = nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(2048, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            self.deconv_block4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block5 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            print("That")

        # Regressor
        self.fc_dim_reduce = nn.Linear(56 * 56 * 32, 1024)
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

        # Initialize Weights
        #init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block,
        #                self.fc_dim_reduce, self.fc_trans, self.fc_rot]
        init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block1,self.conv_block2,self.conv_block3,
                        self.fc_dim_reduce]#, self.fc_trans, self.fc_rot]

        for module in init_modules:
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward_one(self, x):
        # Conv
        x = self.init_block(x) #channels=64
        x_res1 = self.res_block1(x) #channels=256
        x_res2 = self.res_block2(x_res1) #channels=512
        x_res3 = self.res_block3(x_res2) #channels=1024
        x_res4 = self.res_block4(x_res3) #channels=2048

        #print('shapes x = {}, x_res1 = {}, x_res2 = {}, x_res3 = {}, x_res4 = {]'.format(x.shape(), x_res1.shape(), x_res2.shape(), x_res3.shape(), x_res4.shape()))
        #print('shapes x = {}, x_res1 = {}, x_res2 = {}, x_res3 = {}, x_res4 = {}'.format(x.size(), x_res1.size(), x_res2.size(), x_res3.size(), x_res4.size()))

        # Deconv
        x_deconv1 = self.deconv_block1(x_res4)
        if self.sum_mode:
            x_deconv1 = x_res3 + x_deconv1
        else:
            x_deconv1 = torch.cat((x_res3, x_deconv1), dim=1)

        x_deconv2 = self.deconv_block2(x_deconv1)
        if self.sum_mode:
            x_deconv2 = x_res2 + x_deconv2
        else:
            x_deconv2 = torch.cat((x_res2, x_deconv2), dim=1)

        x_deconv3 = self.deconv_block3(x_deconv2)
        if self.sum_mode:
            x_deconv3 = x_res1 + x_deconv3
        else:
            x_deconv3 = torch.cat((x_res1, x_deconv3), dim=1)

        x_conv = self.conv_block1(x_deconv3)
        x_conv = self.conv_block2(x_conv)
        x_conv = self.conv_block3(x_conv)
        #print("shape of output feature : {}".format(x_conv.shape))
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
        rot = self.FinalLayer_rot(out)
        trans = self.FinalLayer_tra(out)
        #return rot, trans, x_conv1, x_conv2, x1_, x2_
        return rot, trans, x_conv1, x_conv2
        #return rot, trans