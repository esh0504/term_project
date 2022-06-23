
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import torchvision
import glob
from inplace_abn import InPlaceABN
import re
import warnings
warnings.filterwarnings("ignore")
#lgr = logging.getLogger(__name__)

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class CostRegNet(nn.Module):
    def __init__(self, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))

        x = conv4 + self.conv7(x) # x = (7*batch_size, 32, 4, 14, 14)
        del conv4
        x = conv2 + self.conv9(x) # x = (7*batch_size, 16, 8, 28, 28)
        del conv2
        x = conv0 + self.conv11(x) # x = (7*batch_size, 8, 16, 56, 56)
        del conv0
        x = self.prob(x) # x = (7*batch_size, 1, 16, 56, 56)
        
        return x


class HourglassNet(nn.Module):
    def __init__(self, base_model, sum_mode=False, dropout_rate=0.0, bayesian=False):
        super(HourglassNet, self).__init__()
        self.cost_reg = CostRegNet()
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

            # print("This")
        else:
            self.deconv_block1 = nn.ConvTranspose2d(2048, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(2048, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),bias=False, output_padding=1)
            self.deconv_block4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block5 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # print("That")

        
        self.volume1 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.volume2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.volume3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.volume4 = nn.Conv2d(32, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.volume5 = nn.Conv2d(32, 32, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4), bias=False)
        self.volume6 = nn.Conv2d(32, 32, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5), bias=False)
        self.volume7 = nn.Conv2d(32, 32, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), bias=False)
        self.volume8 = nn.Conv2d(32, 32, kernel_size=(15, 15), stride=(1, 1), padding=(7, 7), bias=False)
        self.volume9 = nn.Conv2d(32, 32, kernel_size=(17, 17), stride=(1, 1), padding=(8, 8), bias=False)
        self.volume10 = nn.Conv2d(32, 32, kernel_size=(19, 19), stride=(1, 1), padding=(9, 9), bias=False)
        self.volume11 = nn.Conv2d(32, 32, kernel_size=(21, 21), stride=(1, 1), padding=(10, 10), bias=False)
        self.volume12 = nn.Conv2d(32, 32, kernel_size=(23, 23), stride=(1, 1), padding=(11, 11), bias=False)
        self.volume13 = nn.Conv2d(32, 32, kernel_size=(25, 25), stride=(1, 1), padding=(12, 12), bias=False)
        self.volume14 = nn.Conv2d(32, 32, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), bias=False)
        self.volume15 = nn.Conv2d(32, 32, kernel_size=(29, 29), stride=(1, 1), padding=(14, 14), bias=False)
        
        
        
        # Regressor
        self.fc_dim_reduce = nn.Linear(56 * 56 * 16, 1024)
        #self.fc_trans = nn.Linear(1024, 3)
        #self.fc_rot = nn.Linear(1024, 4)

        self.FinalLayer_rot = nn.Sequential(
            # nn.Linear(4096 * 3, 1024),
            #nn.Linear(1024 * 3, 1024),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            #nn.PReLU(),
            nn.Linear(1024, 256),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            nn.PReLU(),
            nn.Linear(256, 4)
        )

        self.FinalLayer_tra = nn.Sequential(
            # nn.Linear(4096 * 3, 1024),
            #nn.Linear(1024*3, 1024),
            #nn.ReLU(),
            #nn.Tanh(),
            #nn.Tanhshrink(),
            #nn.PReLU(),
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
        del x_deconv1
        del x_deconv2
        del x_deconv3
        x_conv = self.conv_block2(x_conv)
        x_conv = self.conv_block3(x_conv)
        
        #print("shape of output feature : {}".format(x_conv.shape))
        x_volume =  x_conv.unsqueeze(2).repeat(1, 1, 16, 1, 1)
        volume1 = self.volume1(x_conv)
        x_volume[:,:,1] = volume1
        del volume1
        volume2 = self.volume2(x_conv)
        x_volume[:,:,2] = volume2
        del volume2
        volume3 = self.volume3(x_conv)
        x_volume[:,:,3] = volume3
        del volume3
        volume4 = self.volume4(x_conv)
        x_volume[:,:,4] = volume4
        del volume4
        volume5 = self.volume5(x_conv)
        x_volume[:,:,5] = volume5
        del volume5
        volume6 = self.volume6(x_conv)
        x_volume[:,:,6] = volume6
        del volume6
        volume7 = self.volume7(x_conv)
        x_volume[:,:,7] = volume7
        del volume7
        volume8 = self.volume8(x_conv)
        x_volume[:,:,8] = volume8
        del volume8
        volume9 = self.volume9(x_conv)
        x_volume[:,:,9] = volume9
        del volume9
        volume10 = self.volume10(x_conv)
        x_volume[:,:,10] = volume10
        del volume10
        volume11 = self.volume11(x_conv)
        x_volume[:,:,11] = volume11
        del volume11
        volume12 = self.volume12(x_conv)
        x_volume[:,:,12] = volume12
        del volume12
        volume13 = self.volume13(x_conv)
        x_volume[:,:,13] = volume13
        del volume13
        volume14 = self.volume14(x_conv)
        x_volume[:,:,14] = volume14
        del volume14
        volume15 = self.volume15(x_conv)
        x_volume[:,:,15] = volume15
        del volume15
        
        x_volume = x_volume ** 2
        #x_linear = F.relu(x_linear)
        #x_linear = F.tanh(x_linear) # changed Relu to Tanh
        #x_linear = F.tanhshrink(x_linear)

        # x_linear = F.leaky_relu(x_linear, negative_slope=0.2, inplace=False)
        #print("shape of x_linear : {}".format(x_linear.shape))#

        # dropout_on = self.training or self.bayesian
        # if self.dropout_rate > 0:
        #     x_linear = F.dropout(x_linear, p=self.dropout_rate, training=dropout_on)
            #x_linear = torch.mul(x_linear, self.dropout_rate)
            #print("shape of x_linear : {}".format(x_linear.shape))

        #trans = self.fc_trans(x_linear)
        #rot = self.fc_rot(x_linear)

        #return trans, rot
        return x_conv, x_volume
        #return x_linear

    def forward(self, x1, x2):
        x_conv1, ref_volume = self.forward_one(x1)
        del x_conv1
        #x1_ = self.forward_one(x1)
        x_conv2, src_volume = self.forward_one(x2)
        #x2_ = self.forward_one(x2)
        del x_conv2
        
        
        diff_volume = torch.abs(ref_volume - src_volume)
        del ref_volume, src_volume
        
        diff = self.cost_reg(diff_volume).squeeze(1)
        del diff_volume
        
        
        #print('size of diff={}, x2_ = {}, x1_={}'.format(diff.shape, x2_.shape, x1_.shape))
        diff = diff.view(diff.size(0), -1)
        out = self.fc_dim_reduce(diff)
        del diff
        rot = self.FinalLayer_rot(out)
        trans = self.FinalLayer_tra(out)
        #return rot, trans, x_conv1, x_conv2, x1_, x2_
        # print(rot.shape, trans.shape, x_conv1.shape, x_conv2.shape)
        return rot, trans, 1, 2
        #return rot, trans