
import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset  # For custom data-sets



#base_imgDir_train = '/media/ubuntu/dbfa9ab7-472f-4aba-8300-c3f65e574665/DTU_Dataset/mvs_training/dtu/Rectified/train/scan'
#base_imgDir_test =  '/media/ubuntu/0ad7b605-b3e4-4967-b2c6-a5cedd7cd273/Downloads/DTU_MVSData/mvs_training/dtu/Rectified/scan'

class datasource(object):
    def __init__(self, images1, poses_r, poses_t, idx, max_size, categ):
        self.images1 = images1
        self.poses_r = poses_r
        self.poses_t = poses_t
        self.max_size = max_size
        self.idx = idx
        self.pos = 0
        self.categ = categ

class datasource_test(object):
    def __init__(self, images1, poses_r, poses_t, idx, max_size, categ):
        self.images1 = images1
        self.poses_r = poses_r
        self.poses_t = poses_t
        self.max_size = max_size
        self.idx = idx
        self.pos = 0
        self.categ = categ


class CustomDataset2(Dataset):
    def __init__(self, image_paths, img_size, img_crop, train=True):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transform(image)
        return image

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)


class CustomDataset_poses(Dataset):
    def __init__(self, image_paths, train=True):
        self.image_paths = image_paths

    def __getitem__(self, index):
        pose = torch.FloatTensor(self.image_paths[index])
        return (pose)

    def __len__(self):  # return count of sample we have
        return len(self.image_paths)

    def flaotTensorToImage(img, mean=0, std=1):
        """convert a tensor to an image"""
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = np.squeeze(img) * 255
        img = img.astype(np.uint8)
        return img

def get_data(base_imgDir_train, mode = 'train'):
    poses_r = []
    poses_t = []
    images1 = []

    categ = []

    with open('./Hourglass_Training/train_mvs.txt') as f:
        for line in f:
            imgFiledId1, categoryId, q1, q2, q3, q4, x, y, z = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            q1 = float(q1)
            q2 = float(q2)
            q3 = float(q3)
            q4 = float(q4)
            poses_r.append((q1 ,q2 ,q3 ,q4))
            poses_t.append((x ,y ,z))
            categ.append(int(categoryId))
            imgFiledId1 = '0' + imgFiledId1 if len(imgFiledId1)==1 else imgFiledId1
            images1.append(base_imgDir_train + categoryId + '_train/' +'rect_0' + imgFiledId1 + '_3_r5000.png')

    max_size = len(poses_r)
    indices = list(range(max_size))
    random.shuffle(indices)
    return datasource(images1, poses_r, poses_t, indices, max_size, categ)

'''def gen_data_batch(source,batch_size):
    image1_batch = []
    pose_x_batch = []
    pose_q_batch = []
    for i in range(batch_size):
        pos = i + source.pos
        pose_x = source.poses_r[source.idx[pos]]
        pose_q = source.poses_t[source.idx[pos]]
        image1_batch.append(source.images1[source.idx[pos]])
        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)

    source.pos += i
    if source.pos + i > source.max_size:
        source.pos = 0
    return image1_batch, np.array(pose_x_batch), np.array(pose_q_batch) '''



def get_data_test(base_imgDir_test, mode='test'):
    poses_r = []
    poses_t = []
    images1 = []
    categ = []

    with open('./Hourglass_Training/test_mvs.txt') as f:
        next(f)  # skip the 3 header lines
        next(f)
        next(f)
        for line in f:
            imgFiledId1, categoryId, q1, q2, q3, q4, x, y, z = line.split()
            x = float(x)
            y = float(y)
            z = float(z)
            q1 = float(q1)
            q2 = float(q2)
            q3 = float(q3)
            q4 = float(q4)
            poses_r.append((q1, q2, q3, q4))
            poses_t.append((x, y, z))
            categ.append(int(categoryId))
            imgFiledId1 = '0' + imgFiledId1 if len(imgFiledId1) == 1 else imgFiledId1
            images1.append(base_imgDir_test + categoryId + '_train/' + 'rect_0' + imgFiledId1 + '_3_r5000.png')

    max_size = len(poses_r)
    indices = list(range(max_size))
    random.shuffle(indices)
    return datasource_test(images1, poses_r, poses_t, indices, max_size, categ)

'''def gen_data_batch_test(source, batch_size):
    image1_batch = []
    pose_x_batch = []
    pose_q_batch = []
    for i in range(batch_size):
        pos = i + source.pos
        pose_x = source.poses_r[source.idx[pos]]
        pose_q = source.poses_t[source.idx[pos]]
        image1_batch.append(source.images1[source.idx[pos]])

        pose_x_batch.append(pose_x)
        pose_q_batch.append(pose_q)

    source.pos += i
    if source.pos + i > source.max_size:
        source.pos = 0
    return image1_batch, np.array(pose_x_batch), np.array(pose_q_batch)
    '''