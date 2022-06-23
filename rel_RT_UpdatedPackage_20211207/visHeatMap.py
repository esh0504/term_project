import numpy as np
import cv2
import argparse
from torchvision import models, transforms
from torch.nn import functional as F
from torch import topk

# Combined imports
import os
from opt import get_opts
import torch
import torch.nn as nn

# RTNet imports
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
import torch.optim as optim
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import random
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import cv2
import glob
import logging
import re
from torch import sigmoid
# from sklearn.preprocessing import MinMaxScaler
from torch.nn.init import xavier_uniform_, zeros_
import warnings

warnings.filterwarnings("ignore")
lgr = logging.getLogger(__name__)
from logger import Logger as Log
import datetime

# MVSNET imports
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# optimizer, scheduler, visualization
from utils import *
from utils_zahid import *

# losses
from losses import loss_dict

# data loaders
# from datasets.dtu_umar import *
from datasets.dtu_both_relative import DTUDataset
from datasets.dtu_both_relative6 import DTUDataset # --> new conversion (relRT25)

# metrics
from metrics import *

# import writing results
from torchvision.utils import save_image

# models
from models.hourglass_Zahid import HourglassNet
from models.hourglass_Zahidv7 import HourglassNet # for RTNet_Zahidv9 we are using new network, hourglass_zahidv7 (relRT25)
#from models.StackedHourglass_Zahid.py import HourGlassModule as HourglassNet
from models.hourglass_Zahidv6_Res50_conv_disp import HourglassNet
from models.hourglass_Zahidv6_Res50 import HourglassNet
from models.mvsnet import MVSNet
from inplace_abn import InPlaceABN


torch.backends.cudnn.benchmark = True  # this increases training speed by 5x


args = get_opts()

base_model_RT = models.resnet50(pretrained=True)
if args.base_model == 'wide_resnet50':
    base_model_RT = models.wide_resnet50_2(pretrained=True)
if args.base_model == 'resnext50':
    base_model_RT = models.resnext50_32x4d(pretrained=True)
if args.base_model == 'resnet34':
    base_model_RT = models.resnet34(pretrained=True)




outdir = "./ckpts/esh_RT/" + "visHeatMat" + "/"

if not os.path.exists(outdir):
    os.mkdir(outdir)

outdir = outdir + args.base_model + "_" + args.loss_method +"_" + str(args.loss_feat) +"/"

if not os.path.exists(outdir):
    os.mkdir(outdir)
    
outdir = outdir + "epoch_" + str(args.visual_epoch) + "/"

if not os.path.exists(outdir):
    os.mkdir(outdir)

# outdir = "./ckpts/Zahid/exp1/"

img_size = 256
img_crop = 224


# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    #weight_softmax = weight_softmax.reshape((512, 2048))
    print('weight shape: ',weight_softmax.shape)
    print('feature shape: ',feature_conv.shape)
    for idx in range(0,class_idx):
        cam = (weight_softmax[idx]).dot(feature_conv.reshape((nc, w*h)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.2
        # put class label text on the result
        cv2.putText(result, str(i), (20, 40), 1, 2, (0, 255, 0), 2)
        #cv2.imshow('CAM', result/255.)
        #cv2.waitKey(0)
        cv2.imwrite(f"{outdir}/scan12_{save_name}1.jpg", result)

model_RT_relative = HourglassNet(base_model_RT, True, 0.3, True)

modelRT_load_path = 'ckpts/ckpts_23.model'
checkpoint = torch.load(modelRT_load_path)
model_RT_relative.load_state_dict(checkpoint)
model_RT_relative = model_RT_relative.cuda()
print('Loading model ckpt done !!')


# define the transforms, resize => tensor => normalize
transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

fn1 = '/home/gpuadmin/Seungho/datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/Rectified/scan12_train/rect_001_0_r5000.png'
fn2 = '/home/gpuadmin/Seungho/datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/Rectified/scan12_train/rect_010_0_r5000.png'

image1 = cv2.imread(fn1).astype('uint8')
image2 = cv2.imread(fn2).astype('uint8')
image1_ = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_ = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

height, width, _ = image1.shape

image1_tensor = transforms(image1_)
image2_tensor = transforms(image2_)

image1_tensor = image1_tensor.unsqueeze(0)
image2_tensor = image2_tensor.unsqueeze(0)


model_RT_relative.eval()

params = list(model_RT_relative.parameters())

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model_RT_relative._modules.get('res_block4')[2].relu.register_forward_hook(hook_feature)
weight_softmax = np.squeeze(params[-27].detach().cpu().data.numpy()) # last convolutional layer with 32 channel output

pre_rot, pre_tra, x_conv1, x_conv2 = model_RT_relative(image1_tensor.cuda(), image2_tensor.cuda())
pre_rot = F.normalize(pre_rot, p=2, dim=1)

class_idx = 2048 #number of channels in the last conv layer

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
# file name to save the resulting CAM image with
save_name = "rect_001_0_r5000"
# show and save the results
show_cam(CAMs, width, height, image1, save_name)

save_name = "rect_010_0_r5000"
# show and save the results
show_cam(CAMs, width, height, image2, save_name)




'''
def load_synset_classes(file_path):
    # load the synset text file for labels
    all_classes = []
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
        labels = [line.split('\n') for line in all_lines]
        for label_list in labels:
            current_class = [name.split(',') for name in label_list][0][0][10:]
            all_classes.append(current_class)
    return all_classes
# get all the classes in a list
all_classes = load_synset_classes('LOC_synset_mapping.txt')

# read and visualize the image
image = cv2.imread(args['input'])
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# load the model
model = models.resnet18(pretrained=True).eval()
# hook the feature extractor
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('layer4').register_forward_hook(hook_feature)
# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

# define the transforms, resize => tensor => normalize
transforms = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

# apply the image transforms
image_tensor = transforms(image)
# add batch dimension
image_tensor = image_tensor.unsqueeze(0)
# forward pass through model
outputs = model(image_tensor)
# get the softmax probabilities
probs = F.softmax(outputs).data.squeeze()
# get the class indices of top k probabilities
class_idx = topk(probs, 1)[1].int()

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
# file name to save the resulting CAM image with
save_name = f"{args['input'].split('/')[-1].split('.')[0]}"
# show and save the results
show_cam(CAMs, width, height, orig_image, class_idx, all_classes, save_name)

'''

'''
model_weights = []  # we will save the conv layer weights in this list
conv_layers = []  # we will save the 49 conv layers in this list
# get all the model children as list
model_children = list(model_RT_relative.children())

# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective weights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            #for child in model_children[i][j].children():
            #for child in model_children[i][j]:
            if type(model_children[i][j]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i][j].weight)
                conv_layers.append(model_children[i][j])
print(f"Total convolutional layers: {counter}")

# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

# visualize the first conv layer filters
print("size of model_weights : {}".format(model_weights[0].shape))
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model_weights[1]):
    plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    print("size of filter : {}".format(filter.shape))
    plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
    plt.axis('off')
    plt.savefig('./filters/filter2.png')
plt.show()

for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64:  # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter.detach().cpu(), cmap='gray')
        plt.axis("off")
    print(f'Saving layer {num_layer} feature maps...')
    plt.savefig(f'./filters/layer_{num_layer}.png')

quit() '''