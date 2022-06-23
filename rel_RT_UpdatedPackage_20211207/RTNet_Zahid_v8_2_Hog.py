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
#from datasets.dtu_both_relative7 import DTUDataset # --> old conversion
#from datasets.dtu_both_relative6 import DTUDataset # --> new conversion
from datasets.dtu_both_relative8_Hog2 import DTUDataset # --> new conversion, includes reading KPS files, also computes HoG feature in the dataloader. Updated dataloader to merge the HogImage in Blue channel of the image

# metrics
from metrics import *

# import writing results
from torchvision.utils import save_image

# models
#from models.hourglass_Zahidv6_HoG import HourglassNet # (this version failed to converge, loss does not reduce than 20,000
# from models.hourglass_Zahidv6_HoG2 import HourglassNet  # (this version also failed to converge, loss does not reduce than 20,000
from models.hourglass_Zahidv6 import HourglassNet  # trying merging the image with HOG image, and using same old network
#from models.hourglass_Zahidv6 import HourglassNet
#from models.hourglass_Zahidv6_Res50 import HourglassNet
from models.mvsnet import MVSNet
from inplace_abn import InPlaceABN


torch.backends.cudnn.benchmark = True  # this increases training speed by 5x

args = get_opts()
base_model_RT = models.resnet34(pretrained=True)
#base_model_RT = models.resnet50(pretrained=True)

numEpochs = args.num_epochs  # 1000
batch_size = args.batch_size  # 8

outdir = "./ckpts/Zahid_RT/" + args.exp_name + "/"

if not os.path.exists(outdir):
    os.mkdir(outdir)

outdir = "./ckpts/Zahid_RT/" + args.exp_name + "/" + "trainResults"

# outdir = "./ckpts/Zahid/exp1/"

img_size = 256
img_crop = 224

# intrinsics = np.array([[2892.33, 0, 823.205],[0, 2883.18, 619.071],[0, 0, 1]], dtype='double')
intrinsics, depth_min, depth_interval = read_intrinsics_file(args.intrinsic_file_path, args.interval_scale)

LogDir = "./ckpts/Zahid_RT/" + args.exp_name + "/trainLog_" + str(datetime.datetime.now()) + ".log"
Log.initialize(LogDir)

Log.i('Arguments:')
for k, v in vars(args).items():
    Log.i('\t{} = {}'.format(k, v))

myTransforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_crop),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

unpreprocess = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

unpreprocess2_RT = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    ])

#sift = cv2.xfeatures2d.SIFT_create() #initial sift detector


def save_models(model, epoch, save_dir):
    torch.save(model.state_dict(), "{}/ckpts_{}.model".format(save_dir, epoch))
    print("Checkpoint saved")
    Log.i("{}/ckpts_{}.model".format(save_dir, epoch))


def perc_loss(pred, gt):
    '''print('pred = {}'.format(pred[:, 0]))
    print('gt = {}'.format(gt[:, 0]))
    print('div = {}'.format(torch.div((pred[:, 0] - gt[:, 0].cuda()) , gt[:, 0].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 0] - gt[:, 0].cuda()) , gt[:, 0].cuda())))))

    print('pred = {}'.format(pred[:, 1]))
    print('gt = {}'.format(gt[:, 1]))
    print('div = {}'.format(torch.div((pred[:, 1] - gt[:, 1].cuda()), gt[:, 1].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 1] - gt[:, 1].cuda()), gt[:, 1].cuda())))))

    print('pred = {}'.format(pred[:, 2]))
    print('gt = {}'.format(gt[:, 2]))
    print('div = {}'.format(torch.div((pred[:, 2] - gt[:, 2].cuda()), gt[:, 2].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 2] - gt[:, 2].cuda()), gt[:, 2].cuda())))))

    print('pred = {}'.format(pred[:, 3]))
    print('gt = {}'.format(gt[:, 3]))
    print('div = {}'.format(torch.div((pred[:, 3] - gt[:, 3].cuda()), gt[:, 3].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 3] - gt[:, 3].cuda()), gt[:, 3].cuda())))))

    print('pred = {}'.format(pred[:, 4]))
    print('gt = {}'.format(gt[:, 4]))
    print('div = {}'.format(torch.div((pred[:, 4] - gt[:, 4].cuda()), gt[:, 4].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 4] - gt[:, 4].cuda()), gt[:, 4].cuda())))))

    print('pred = {}'.format(pred[:, 5]))
    print('gt = {}'.format(gt[:, 5]))
    print('div = {}'.format(torch.div((pred[:, 5] - gt[:, 5].cuda()), gt[:, 5].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 5] - gt[:, 5].cuda()), gt[:, 5].cuda())))))

    print('pred = {}'.format(pred[:, 6]))
    print('gt = {}'.format(gt[:, 6]))
    print('div = {}'.format(torch.div((pred[:, 6] - gt[:, 6].cuda()), gt[:, 6].cuda())))
    print('loss = {}'.format(100 * torch.sum(abs(torch.div((pred[:, 6] - gt[:, 6].cuda()), gt[:, 6].cuda())))))


    print('overall loss = {}'.format(100 * (torch.sum(abs(torch.div((pred[:, 0] - gt[:, 0].cuda()) , gt[:, 0].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 1] - gt[:, 1].cuda()) , gt[:, 1].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 2] - gt[:, 2].cuda()) , gt[:, 2].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 3] - gt[:, 3].cuda()) , gt[:, 3].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 4] - gt[:, 4].cuda()) , gt[:, 4].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 5] - gt[:, 5].cuda()) , gt[:, 5].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 6] - gt[:, 6].cuda()) , gt[:, 6].cuda())))) ))

    quit()'''
    return 100 * (torch.sum(abs(torch.div((pred[:, 0] - gt[:, 0].cuda()) , gt[:, 0].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 1] - gt[:, 1].cuda()) , gt[:, 1].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 2] - gt[:, 2].cuda()) , gt[:, 2].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 3] - gt[:, 3].cuda()) , gt[:, 3].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 4] - gt[:, 4].cuda()) , gt[:, 4].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 5] - gt[:, 5].cuda()) , gt[:, 5].cuda())))\
    + torch.sum(abs(torch.div((pred[:, 6] - gt[:, 6].cuda()) , gt[:, 6].cuda()))))

def test_dataloader():
    test_dataset = DTUDataset(root_dir=args.root_dir,
                              split='test',
                              n_views=args.n_views,
                              n_depths=args.n_depths,
                              interval_scale=args.interval_scale,
                              img_size=args.img_size,
                              img_crop=args.img_crop,
                              intrinsic_file_path=args.intrinsic_file_path
                              )
    if args.num_gpus > 1:
        sampler = DistributedSampler(test_dataset)
    else:
        sampler = None
    return DataLoader(test_dataset,
                      shuffle=False,  # (sampler is None),
                      sampler=sampler,
                      num_workers=4,
                      batch_size=args.batch_size,
                      pin_memory=True)

def train_dataloader():
    train_dataset = DTUDataset(root_dir=args.root_dir,
                               split='train',
                               n_views=args.n_views,
                               n_depths=args.n_depths,
                               interval_scale=args.interval_scale,
                               img_size=args.img_size,
                               img_crop=args.img_crop,
                               intrinsic_file_path=args.intrinsic_file_path
                               )
    if args.num_gpus > 1:
        sampler = DistributedSampler(train_dataset)
    else:
        sampler = None
    return DataLoader(train_dataset,
                      shuffle=True,  # (sampler is None),
                      sampler=sampler,
                      num_workers=4,
                      batch_size=args.batch_size,
                      pin_memory=True)


test_dataloader_ = test_dataloader()
train_dataloader_ = train_dataloader()


def configure_optimizers_MVSNet(model_MVSNet):
    optimizer_MVSNet = get_optimizer(args, model_MVSNet)
    scheduler_MVSNet = get_scheduler(args, optimizer_MVSNet)

    # return [optimizer_MVSNet], [scheduler]
    return optimizer_MVSNet, scheduler_MVSNet


def configure_optimizers_RTNet(model_RtNet):
    #optim_params = [{'params': model_RtNet.parameters(), 'lr': 0.00001}]
    #optimizer_RTNet = optim.Adam(optim_params, weight_decay=0.0005)
    # scheduler_RTNet = get_scheduler(args, optimizer_RTNet)

    optimizer_RTNet = get_optimizer(args, model_RtNet)
    scheduler_RTNet = get_scheduler(args, optimizer_RTNet)

    return optimizer_RTNet, scheduler_RTNet


def train_RTNet(model_RT_relative, optimizer_RTNet, schedular_RTNet, numEpochs, model_save_path):
    if args.trainRT_net:
        model_RT_relative.train()
        # criterion_RT = nn.MSELoss(reduction='mean')
        criterion_RT = nn.L1Loss(reduction='sum')
        print("Training RT Net ..")
    else:
        model_RT_relative.eval()
        print("RT Net is used as RT predictor only ..")

    set = '/trainset/'

    w1 = 200
    w2 = 1

    old_loss = 150000

    for epoch in range(args.start_epoch, numEpochs + 1):
        print("Epoch : %d \n" % epoch)
        if not os.path.exists('./' + outdir + '/epoch_' + str(epoch)):
            os.mkdir('./' + outdir + '/epoch_' + str(epoch))

        for batch_idx, (
                #ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, depth, kps_, HogFeat) in enumerate(
                #train_dataloader_):
                ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, depth, kps_) in enumerate(train_dataloader_):
            #print("Filename : {} \n".format(filename_to_save_))
            #print("rel_quat_ : {} \n".format(rel_quat_))

            # images1 = Variable(images1.cuda(async=True))
            # poses_r = Variable(poses_r.cuda(async=True))
            # poses_t = Variable(poses_t.cuda(async=True))
            # gt_poses = torch.cat((poses_r, poses_t), dim=1)

            # pos_out, ori_out = model(images1)
            B, V, C, H, W = ref_imgs.shape  # Zahid for RT B - batch size, V - no of views
            #print("B={}, V={}, C={}, H={}, W={}".format(B, V, C, H, W))


            B_, V_, C_, H_, W_ = src_imgs.shape  # Zahid for RT B - batch size, V - no of views
            #print("B_={}, V_={}, C_={}, H_={}, W_={}".format(B_, V_, C_, H_, W_))


            src_imgs = src_imgs.reshape(B_ * V_, 3, H_, W_)  # --> pass the view images as batch images to RT network
            #img_ = (src_imgs[:, :, ::2, ::2]).cpu()  # batch 0, all ref images, 1/4 scale
            #save_image(img_, outdir + '/epoch_' + str(epoch) + "/" + "src.png")
            #save_image(img_, "src.png")

            ref_imgs = ref_imgs.reshape(B * V, 3, H, W)  # --> pass the view images as batch images to RT network
            # print("B={}, V={}, C={}, H={}, W={}".format(B, V, C, H, W))
            #img_ = (ref_imgs[ :, :, ::2, ::2]).cpu()  # batch 0, all ref images, 1/4 scale
            #save_image(img_, outdir + '/epoch_' + str(epoch) + "/" + "ref.png")
            #save_image(img_, "ref.png")
            #quit()

            # print("size of gt_qua = {}, gt_qua = {}".format((gt_quaternion[0].shape), torch.cat(gt_quaternion[0], gt_quaternion[1], gt_quaternion[0])))
            rel_quat_ = rel_quat_.reshape(B * V, 7)

            #HogFeat = HogFeat.reshape(B * V, 52488)
            #HogFeat = HogFeat.reshape(B * V, 12168)

            #print('filename_to_save_src_: ', filename_to_save_src)
            #print('filename_to_save_ref_: ', filename_to_save_ref)
            #print("size of gt_qua = {}, gt_qua = {}".format((rel_quat_.shape), rel_quat_))
            #print("Size of HoG_features : {}".format(HogFeat.shape))


            optimizer_RTNet.zero_grad()
            #rot_out, tra_out, _, _ = model_RT_relative(ref_imgs.cuda(), src_imgs.cuda(), HogFeat.cuda())
            rot_out, tra_out, _, _ = model_RT_relative(ref_imgs.cuda(), src_imgs.cuda())

            #print("size of pos_out = {}, ori_out = {}, pos_out={}, ori_out={}".format(tra_out.shape, rot_out.shape, tra_out, rot_out))
            #print("size of pos_out={}, ori_out={}".format(tra_out,rot_out))

            rot_out = F.normalize(rot_out, p=2, dim=1) #--> only for relRT18

            pred = torch.cat((rot_out, tra_out), dim=1)

            #kps_reshaped = (kps_.cpu().detach().numpy()).reshape(B * V, 4, 4)
            # kps_reshaped = (kps_.cpu().detach().numpy()).reshape(B * V, 20, 4) #-->relRT34
            # kps_reshaped = (kps_.cpu().detach().numpy()).reshape(B * V, 3, 4)  # -->relRT36
            kps_reshaped = (kps_.cpu().detach().numpy()).reshape(B * V, 2, 4)  # -->relRT36 funetine after 100 epochs
            '''
            print('filename_to_save_src_: ', len(filename_to_save_src), len(filename_to_save_src[0]))
            print('filename_to_save_ref_: ', len(filename_to_save_ref), len(filename_to_save_ref[0]))
            print(' KPS size :', kps.shape)


            for i in range(B_):
                for j in range(V_):
                    print('filename_to_save_src_: ', filename_to_save_src[j][i])
                    print('filename_to_save_ref_: ', filename_to_save_ref[0][i])
                    print('kps[{}, {}] : {}'.format(i, j, kps[i][j]))
                    print('rel_quat ',rel_quat_[i*V_ + j])

            quit()'''

            loss_feat = 0
            cnt = 0
            for i in range(B_ * V_):
                # here we used a fixed points, rather than taking from SIFT
                points1 = kps_reshaped[i,:,:2]
                points2_proj_GT = kps_reshaped[i, :, 2:4]
                #print('points1[{}] : {}'.format(i, points1))
                #print('points2_gt[{}] : {}'.format(i, points2_proj_GT))
                '''points1 = np.array(
                    [[10, 35], [11, 45], [18, 40], [25, 50], [25, 20], [50, 20], [70, 20], [90, 20], [110, 20], \
                     [140, 50], [25, 70], [25, 80], [70, 70], [110, 110]])'''

                '''points2_proj_GT = inverse_project(intrinsics, rel_quat_[i, :4], rel_quat_[i, 4:], depth[math.floor(i / V_)],
                                                  points1)'''
                points2_proj_pre = inverse_project3(intrinsics, pred[i, :4].cpu().detach(), pred[i, 4:].cpu().detach(),
                                                   depth[math.floor(i / V_)],
                                                   points1)


                '''fp = open("points1"+ str(i) +".txt", "w")
                fp1 = open("pointsGT"+str(i)+".txt", "w")
                fp2 = open("pointsPre"+str(i)+".txt", "w")
                for j in range(points1.shape[0]):
                    fp.write(str(points1[j][0]) + ' ' + str(points1[j][1]) + '\n')
                    fp1.write(str(points2_proj_GT[j][0]) + ' ' + str(points2_proj_GT[j][1]) + '\n')
                    fp2.write(str(points2_proj_pre[j][0]) + ' ' + str(points2_proj_pre[j][1]) + '\n')
    
                fp.close()
                fp1.close()
                fp2.close()'''

                #print("points1_2D = {}".format(points1))
                #print("points2_proj = {}".format(points2_proj))
                #print("points1_2D size = {}".format(points1.shape))
                #print("points2_2D size = {}".format(points2_proj.shape))

                #quit()

                #points2_proj_GT = np.array(points2_proj_GT)
                points2_proj_pre = np.array(points2_proj_pre)

                #print('points2_proj_pre[{}] : {}'.format(i, points2_proj_pre))
                #print('points2_proj ', points2_proj_pre.shape)

                #mask_ = (mask[math.floor(i / V_)]).cpu().numpy().astype(np.uint8)
                # cv2.imwrite('mask.png',mask_)
                # print("mask size = {}, dtype = {}, ".format(mask_.shape, mask_.dtype))

                #mask_ = cv2.resize(mask_, (160, 128))

                # print('ref point i={}, j={}, = ({},{})'.format(i,j, ref_kps[i][j*stride][1],ref_kps[i][j*stride][0]))
                # print('src point i={}, j={}, =({},{})'.format(i, j, src_kps[i][j * stride][1], src_kps[i][j * stride][0]))
                #loss_feat += np.linalg.norm(points2_proj_GT, points2_proj_pre)

                loss_feat += np.sum(abs(points2_proj_GT[:, 0] - points2_proj_pre[:, 0]) + abs(points2_proj_GT[:, 1] - points2_proj_pre[:, 1]))
                #print("loss feat :", loss_feat)
                #quit()

                '''for j in range(len(points1)):
                    if(points2_proj_GT[j,0]>0 and points2_proj_pre[j,0] >0 and points2_proj_GT[j,1]>0 and points2_proj_pre[j,1]>0 and mask_[points1[j,1], points1[j,0]] > 0\
                        and points2_proj_GT[j, 0] < 160 and points2_proj_pre[j, 0] <160 and points2_proj_GT[j, 1] < 128 and points2_proj_pre[j, 1] < 128 ):
                        loss_feat += abs(points2_proj_GT[j,0] - points2_proj_pre[j,0]) + abs(points2_proj_GT[j,1] - points2_proj_pre[j,1])
                        cnt += 1
                        #print("points2_proj_GT[{},0] - points2_proj_pre[{},0] : {}".format(j,j,points2_proj_GT[j,0] - points2_proj_pre[j,0]))
                        #print("points2_proj_GT[{},1] - points2_proj_pre[{},1] : {}".format(j,j,points2_proj_GT[j, 1] - points2_proj_pre[j, 1]))
                        #print("Loss Feat : {}".format(loss_feat))

                #print("\n\nPer image Loss Feat : {}".format(loss_feat))'''

                #quit()
            '''if(cnt>0):
                loss_feat /= cnt
            else:
                loss_feat = np.array([0])'''
            #loss_feat /= (B_ * V_ * 4)
            #print("Final Loss Feat : {}".format(loss_feat))


            #loss_feat = torch.FloatTensor(loss_feat).cuda()
            #loss_feat = (torch.from_numpy(loss_feat)).cuda()

            #print("Final Loss Feat : {}".format(loss_feat))
            #quit()

            # loss_rtnet = criterion_RT(pred, gt_poses)
            #print("size of pred = {}, pred = {}".format(pred.shape, pred))

            #quit()

            #print('filename_to_save_src_: ', filename_to_save_src)
            #print('filename_to_save_ref_: ', filename_to_save_ref)
            #print("size of gt_qua = {}, gt_qua = {}".format((rel_quat_.shape), rel_quat_))
            #print("Extr = {}".format(Extr_2))
            # loss_rtnet = criterion_RT(pred, gt_quaternion[0].cuda())
            #print("size of pred = {}, pred = {}".format((pred.shape), pred))
            # loss_rtnet = criterion_RT(pred, gt_quaternion.cuda())
            if (0 and args.usePercenLoss):
                #print('Percentageloss ....')
                loss_rtnet = perc_loss(pred, rel_quat_) # --> fintuning the relRT19 experiment, checkpoints will be 100~200
            else:
                loss_rtnet = w1 * criterion_RT(pred[:, 0], rel_quat_[:, 0].cuda()) + w1 * criterion_RT(pred[:, 1],rel_quat_[:,1].cuda()) + w1 * criterion_RT(pred[:, 2], rel_quat_[:, 2].cuda()) + w1 * criterion_RT(pred[:, 3],
                                                                             rel_quat_[:, 3].cuda()) + w2 * criterion_RT(pred[:, 4], rel_quat_[:, 4].cuda()) + w2 * criterion_RT(pred[:, 5],rel_quat_[:, 5].cuda()) + w2 * criterion_RT(
                    pred[:, 6], rel_quat_[:, 6].cuda()) #--> for relRT19, we used this loss as loss_rtnet, and after 100 epochs, we fine tune using perc_loss


            #print("Loss RTNet = {}".format(loss_rtnet))
            #quit()

            # Backpropagate the loss
            (loss_rtnet+loss_feat).backward()
            #(loss_rtnet).backward()

            # Adjust parameters according to the computed gradients
            optimizer_RTNet.step()

            lr = get_learning_rate(optimizer_RTNet)

            print('Epoch : {} out of {}, batch : {} out of {}, RT loss : {}, Feat loss : {}, Total loss : {}, Lr : {}'.format(epoch, args.num_epochs,
                                                                                                batch_idx,
                                                                                                len(train_dataloader_),
                                                                                                loss_rtnet, loss_feat, (loss_rtnet+loss_feat), lr))
            '''Log.i('Epoch : {} out of {}, batch : {} out of {}, Train loss : {}, Lr : {}'.format(epoch, args.num_epochs,
                                                                                                batch_idx,
                                                                                                len(train_dataloader_),
                                                                                                loss_rtnet, lr)) '''

            #schedular_RTNet.step()
            #quit()
        # save_models(model_MVSNet, epoch, model_save_path)
        if(loss_rtnet+loss_feat < old_loss):
            old_loss = loss_rtnet+loss_feat
            save_models(model_RT_relative, epoch, model_save_path)
            Log.i('Saving models at Epoch : {} out of {}, batch : {} out of {}, RT loss : {}, Feat loss : {}, Lr : {}'.format(epoch, args.num_epochs,
                                                                                                    batch_idx,
                                                                                                    len(train_dataloader_),
                                                                                                    loss_rtnet, loss_feat, lr))


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def test_RTNet(model_RT, exp_name, start_epoch):
    avgMSEloss_RTnet = 0
    criterion_RT = nn.L1Loss(reduction='sum')
    model_RT.eval()
    model_RT_relative.apply(apply_dropout)
    print("RT Net is used as RT predictor ..")

    '''
    if args.trainMVSNet or args.trainBoth:
        model_MVSNet.train()
        print("We are going to train MVSNet !")
    else:
        model_MVSNet.eval()
        print("We are going to eval MVSNet !")
    '''

    set = '/trainset/'
    fp_gt = open('./results_prediction/gt_ep' + str(start_epoch) +'_'+ exp_name +'.txt','w')
    fp_pr = open('./results_prediction/pr_ep' + str(start_epoch) +'_'+ exp_name +'.txt','w')

    for batch_idx, (
            ref_imgs, src_imgs, rel_quat, filename_to_save_ref, filename_to_save_src) in enumerate(
        test_dataloader_):
        # print("Filename : {} \n".format(filename_to_save))
        # images1 = Variable(images1.cuda(async=True))
        # poses_r = Variable(poses_r.cuda(async=True))
        # poses_t = Variable(poses_t.cuda(async=True))
        # gt_poses = torch.cat((poses_r, poses_t), dim=1)

        # pos_out, ori_out = model(images1)
        B, V, C, H, W = ref_imgs.shape  # Zahid for RT B - batch size, V - no of views
        # print("B={}, V={}, C={}, H={}, W={}".format(B, V, C, H, W))

        B_, V_, C_, H_, W_ = src_imgs.shape  # Zahid for RT B - batch size, V - no of views
        # print("B_={}, V_={}, C_={}, H_={}, W_={}".format(B_, V_, C_, H_, W_))

        src_imgs = src_imgs.reshape(B_ * V_, 3, H_, W_)  # --> pass the view images as batch images to RT network

        # img_ = (src_imgs[:, :, ::2, ::2]).cpu()  # batch 0, all ref images, 1/4 scale
        # save_image(img_, outdir + '/epoch_' + str(epoch) + "/" + "src.png")

        ref_imgs = ref_imgs.reshape(B * V, 3, H, W)  # --> pass the view images as batch images to RT network
        # print("B={}, V={}, C={}, H={}, W={}".format(B, V, C, H, W))

        # img_ = (ref_imgs[ :, :, ::2, ::2]).cpu()  # batch 0, all ref images, 1/4 scale
        # save_image(img_, outdir + '/epoch_' + str(epoch) + "/" + "ref.png")

        with torch.no_grad():
            rot_out, tra_out,_,_ = model_RT_relative(ref_imgs.cuda(), src_imgs.cuda())
        # print("size of pos_out = {}, ori_out = {}, pos_out={}, ori_out={}".format(tra_out.shape, rot_out.shape, tra_out, rot_out))
        # print("size of pos_out={}, ori_out={}".format(tra_out,rot_out))

        with torch.no_grad():
            rot_out = F.normalize(rot_out, p=2, dim=1)

        pred = torch.cat((rot_out, tra_out), dim=1)

        # loss_rtnet = criterion_RT(pred, gt_poses)
        #print("size of pred = {}, pred = {}".format(pred.shape, pred))

        # print("size of gt_qua = {}, gt_qua = {}".format((gt_quaternion[0].shape), torch.cat(gt_quaternion[0], gt_quaternion[1], gt_quaternion[0])))
        rel_quat = rel_quat.reshape(B * V, 7)

        # print("size of gt_qua = {}, gt_qua = {}".format((gt_quaternion.shape), gt_quaternion[0]))
        # loss_rtnet = criterion_RT(pred, gt_quaternion[0].cuda())
        #print("size of gt_qua = {}, gt_qua = {}".format((rel_quat.shape), rel_quat))
        avgMSEloss_RTnet += criterion_RT(pred, rel_quat.cuda())
        #print("Loss = {}".format(avgMSEloss_RTnet))
        #quit()

        rel_quat_ = rel_quat.numpy()
        pred_ = pred.detach().cpu().numpy()

        # print('filename_to_save_src_: ', filename_to_save_src)
        # print('filename_to_save_ref_: ', filename_to_save_ref)
        # print("size of gt_qua = {}, gt_qua = {}".format((rel_quat_.shape), rel_quat_))
        # print("Extr = {}".format(Extr_2))
        # loss_rtnet = criterion_RT(pred, gt_quaternion[0].cuda())
        # print("size of pred = {}, pred = {}".format((pred.shape), pred))
        # loss_rtnet = criterion_RT(pred, gt_quaternion.cuda())

        for i in range(B * V):
            fp_gt.write(str(rel_quat_[i, 0]) + ' ' + str(rel_quat_[i, 1]) + ' ' + str(rel_quat_[i, 2]) + ' ' + str(rel_quat_[i, 3]) + ' ' + str(rel_quat_[i, 4]) + ' ' + str(rel_quat_[i, 5]) + ' ' + str(rel_quat_[i, 6]) + '\n')
            fp_pr.write(str(pred_[i, 0]) + ' ' + str(pred_[i, 1]) + ' ' + str(pred_[i, 2]) + ' ' + str(pred_[i, 3]) + ' ' + str(pred_[i, 4]) + ' ' + str(pred_[i, 5]) + ' ' + str(pred_[i, 6]) + '\n')
        # print("Loss = {}".format(loss_rtnet))
        # quit()

        print('batch : {} out of {}, Test loss : {}'.format(batch_idx, len(test_dataloader_), avgMSEloss_RTnet))

    print('Final avgMSE Loss on Test Set is : {}'.format(avgMSEloss_RTnet / len(test_dataloader_)))
    fp_gt.close()
    fp_pr.close()

if args.trainRT_net:
    # initialize RT Net
    model_RT_relative = HourglassNet(base_model_RT, args.sum_mode, args.dropout_rate, args.bayesian)

    if (1 and args.train_from_scratch == True):
        print('Training relative RT from scratch !', args.train_from_scratch)

    else:
        if not os.path.exists(args.modelRT_load_path):
            print('The pretrained RTNet was not found !')
        else:
            print('Loading RT model from', args.modelRT_load_path)
            checkpoint = torch.load(args.modelRT_load_path)
            # model_RT.load_state_dict(checkpoint['model_state_dict'])
            model_RT_relative.load_state_dict(checkpoint)


    model_RT_relative = model_RT_relative.cuda()

    model_save_path = "./ckpts/Zahid_RT/" + args.exp_name
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # initialize the RTNet optimizer
    optimizer_RTNet, schedular_RTNet = configure_optimizers_RTNet(model_RT_relative)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    train_RTNet(model_RT_relative, optimizer_RTNet, schedular_RTNet, args.num_epochs, model_save_path)


if args.testRT_net:
    # initialize RT Net
    sum_mode = args.sum_mode
    sum_mode = True
    model_RT_relative = HourglassNet(base_model_RT, sum_mode, args.dropout_rate, args.bayesian)

    if not os.path.exists(args.modelRT_load_path):
        print('The pretrained RTNet was not found !')
    else:
        print('Loading RT model from', args.modelRT_load_path)
        checkpoint = torch.load(args.modelRT_load_path)
        # model_RT.load_state_dict(checkpoint['model_state_dict'])
        model_RT_relative.load_state_dict(checkpoint)

        model_RT_relative = model_RT_relative.cuda()

    '''
    # initialize MVSNet
    model_MVSNet = MVSNet(InPlaceABN)

#    if not os.path.exists(args.modelMVSNet_save_path):
    if not os.path.exists(args.ckpt_path):
        print('The pretrained MVSNET was not found !')
    else:
        print('Loading MVSNET model from', args.ckpt_path)
        load_ckpt(model_MVSNet, args.ckpt_path, args.prefixes_to_ignore)

    model_MVSNet = model_MVSNet.cuda()

    '''

    print("Begin testing the trained RTNet")
    test_RTNet(model_RT_relative, args.exp_name, args.start_epoch)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def test_MVSNet(model_RT_relative, model_MVSNet, epoch):
    print('Testing MVSNET started ..')
    model_RT_relative.eval()
    model_RT_relative.apply(apply_dropout)
    model_MVSNet.eval()

    set = '/testset/'

    if not os.path.exists('./' + outdir + '/epoch_' + str(epoch)  + '/' + set):
        os.mkdir('./' + outdir + '/epoch_' + str(epoch)  + '/' + set)

    for batch_idx, (ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, imgs, depth, depth_values, mask, view_ids) in enumerate(test_dataloader_):
        #pos_out, ori_out = model(images1)
        B, V, C, H, W = ref_imgs.shape  # Zahid for RT B - batch size, V - no of views

        #view_ids = view_ids.numpy().astype(np.uint32)
        #print('view ids {}'.format(view_ids))
        # print("B={}, V={}, C={}, H={}, W={}".format(B, V, C, H, W))

        B_, V_, C_, H_, W_ = src_imgs.shape  # Zahid for RT B - batch size, V - no of views
        # print("B_={}, V_={}, C_={}, H_={}, W_={}".format(B_, V_, C_, H_, W_))

        src_imgs = src_imgs.reshape(B_ * V_, 3, H_, W_)  # --> pass the view images as batch images to RT network
        ref_imgs = ref_imgs.reshape(B * V, 3, H, W)  # --> pass the view images as batch images to RT network

        #print('Passing images to RT Net ..')
        with torch.no_grad():
            rot_out, tra_out, _, _ = model_RT_relative(ref_imgs.cuda(), src_imgs.cuda())
        del src_imgs, ref_imgs
        #print("size of pos_out = {}, ori_out = {}, pos_out={}, ori_out={}".format(pos_out.shape, ori_out.shape, pos_out, ori_out))

        rot_out = F.normalize(rot_out, p=2, dim=1) #--> for relRT19

        #print('Passing images to MVSNet ..')
        with torch.no_grad():
            proj_mats = RT_to_ProjectionMartices3(intrinsics, rot_out, tra_out, filename_to_save_ref[0], view_ids, outdir, str(epoch), set, True)

        print("proj Mats = {} , filename_ref = {}".format(proj_mats, filename_to_save_ref[0]))

        filename_to_save_ref = filename_to_save_ref[0]

        with torch.no_grad():
            depth_pred, confidence = model_MVSNet(imgs.cuda(), proj_mats.cuda(), depth_values.cuda())

        img_ = unpreprocess(imgs[0, 0, :, ::4, ::4]).cpu()  # batch 0, ref image, 1/4 scale

        #print('Saving Depth images ..')
        with torch.no_grad():
            if 0: #write outputs
                if os.path.exists('./' + outdir + '/epoch_' + str(epoch) + set):
                    save_image(img_, './' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0])
                    # A1 = (torch.tensor((depth_pred[0]*mask[0]).cpu(), dtype=torch.float32)).cpu().detach().numpy()
                    A1 = (depth_pred[0] * mask[0].cuda()).cpu().numpy().astype(np.float32)
                    write_pfm('./' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0][:-4] + '_init.pfm',
                              A1)
                    del A1

                    # A2 = (torch.tensor((torch.abs((confidence[0]*mask[0]).cpu())), dtype=torch.float32)).cpu().detach().numpy()
                    A2 = (confidence[0] * mask[0].cuda()).cpu().numpy().astype(np.float32)
                    write_pfm('./' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0][:-4] + '_prob.pfm',
                              A2)
                    del A2
                else:
                    os.mkdir('./' + outdir + '/epoch_' + str(epoch) + set)
                    save_image(img_, './' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0])
                    # A1 = (torch.tensor((depth_pred[0]*mask[0]).cpu(), dtype=torch.float32)).cpu().detach().numpy()
                    A1 = (depth_pred[0] * mask[0].cuda()).cpu().numpy().astype(np.float32)
                    write_pfm('./' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0][:-4] + '_init.pfm',
                              A1)
                    del A1

                    # A2 = (torch.tensor((torch.abs((confidence[0]*mask[0]).cpu())), dtype=torch.float32)).cpu().detach().numpy()
                    A2 = (confidence[0] * mask[0].cuda()).cpu().numpy().astype(np.float32)
                    write_pfm('./' + outdir + '/epoch_' + str(epoch) + set + filename_to_save_ref[0][:-4] + '_prob.pfm',
                              A2)
                    del A2
        del img_, depth_pred, confidence, mask
        #quit()

if args.testMVSNet :
    # initialize RT Net
    sum_mode = args.sum_mode
    sum_mode = True
    model_RT_relative = HourglassNet(base_model_RT, sum_mode, args.dropout_rate, args.bayesian)

    if not os.path.exists(args.modelRT_load_path):
        print('The pretrained RTNet was not found !')
    else:
        print('Loading RT model from', args.modelRT_load_path)
        checkpoint = torch.load(args.modelRT_load_path)
        #model_RT.load_state_dict(checkpoint['model_state_dict']) # RT network that UMAR trained requires 'model_state_dict' index
        model_RT_relative.load_state_dict(checkpoint) # RT network that I trained does not require 'model_state_dict' index

        model_RT_relative = model_RT_relative.cuda()


    #initialize MVSNet
    model_MVSNet = MVSNet(InPlaceABN)

    if not os.path.exists(args.modelMVSNet_load_path):
        print('The pretrained MVSNET was not found at {}!'.format(args.modelMVSNet_load_path))
    else:
        print('Loading MVSNET model from', args.modelMVSNet_load_path)
        #checkpoint = torch.load(args.modelMVSNet_save_path)
        #model_MVSNet.load_state_dict(checkpoint['model_state_dict'])
        load_ckpt(model_MVSNet, args.modelMVSNet_load_path, args.prefixes_to_ignore)

    model_MVSNet = model_MVSNet.cuda()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for epoch in range(args.start_epoch, numEpochs + args.start_epoch):
        print("Epoch %d" % epoch)

        if not os.path.exists('./' + outdir + '/epoch_' + str(epoch)):
            os.mkdir('./' + outdir + '/epoch_' + str(epoch))

        test_MVSNet(model_RT_relative, model_MVSNet, epoch)
