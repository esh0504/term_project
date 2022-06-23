import random
from os import path as osp
from collections import defaultdict
from PIL import Image
import torch
import os
import numpy as np


class SevenScenesRelPoseDataset(object):
    def __init__(self, cfg, split='train', transforms=None):
        self.cfg = cfg
        self.split = split
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.t_gt, self.q_gt = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2, t_gt, q_gt = [], [], [], []

        data_params = self.cfg.data_params

        pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id = int(chunks[2])
                fnames1.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[0][1:]))
                fnames2.append(osp.join(data_params.img_dir, self.scenes_dict[scene_id], chunks[1][1:]))

                t_gt.append(torch.FloatTensor([float(chunks[3]), float(chunks[4]), float(chunks[5])]))
                q_gt.append(torch.FloatTensor([float(chunks[6]),
                                               float(chunks[7]),
                                               float(chunks[8]),
                                               float(chunks[9])]))

        return fnames1, fnames2, t_gt, q_gt

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])

        return {'img1': img1,
                'img2': img2,
                't_gt': t_gt,
                'q_gt': q_gt}

    def __len__(self):
        return len(self.fnames1)


class SevenScenesTestDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2 = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2 = [], []

        pairs_txt = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id1 = int(chunks[2])
                scene_id2 = int(chunks[3])
                fnames1.append(osp.join(img_dir, self.scenes_dict[scene_id2], chunks[1][1:]))
                fnames2.append(osp.join(img_dir, self.scenes_dict[scene_id1], chunks[0][1:]))

        return fnames1, fnames2

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return {'img1': img1,
                'img2': img2,
                }

    def __len__(self):
        return len(self.fnames1)





############################ DTU DATASET TRAIN ##################################
class DTUDataset(object):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        #self.scenes_dict = defaultdict(str)
        #for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
        #    self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.R1, self.R2, self.R3, self.T1, self.R4, self.R5, self.R6, self.T2, self.R7, self.R8, self.R9, self.T3 = self._read_pairs_txt()

    def _read_pairs_txt(self):
        #fnames1, fnames2, t_gt, q_gt = [], [], [], []
        fnames1, fnames2, R1, R2, R3, T1, R4, R5, R6, T2, R7, R8, R9, T3 = [], [], [], [], [], [] ,[] ,[] ,[], [], [], [] ,[] ,[]

        #data_params = self.cfg.data_params

        #pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        if self.split == 'train':
            pairs_txt = './datasets/lists/dtu/train_DTU_mat.txt'
        else:
            pairs_txt = './datasets/lists/dtu/test_DTU_mat.txt'
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                #scene_id = int(chunks[2])
                #print((self.root_dir + chunks[0]))
                fnames1.append((self.root_dir + chunks[0]))
                fnames2.append((self.root_dir + chunks[1]))
                R1.append(float(chunks[2]))
                R2.append(float(chunks[3]))
                R3.append(float(chunks[4]))
                T1.append(float(chunks[5]))
                R4.append(float(chunks[6]))
                R5.append(float(chunks[7]))
                R6.append(float(chunks[8]))
                T2.append(float(chunks[9]))
                R7.append(float(chunks[10]))
                R8.append(float(chunks[11]))
                R9.append(float(chunks[12]))
                T3.append(float(chunks[13]))

        return fnames1, fnames2, R1, R2, R3, T1, R4, R5, R6, T2, R7, R8, R9, T3

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        #t_gt = self.t_gt[item]
        #q_gt = self.q_gt[item]
        R1 = self.R1[item]
        R2 = self.R2[item]
        R3 = self.R3[item]
        T1 = self.T1[item]
        R4 = self.R4[item]
        R5 = self.R5[item]
        R6 = self.R6[item]
        T2 = self.T2[item]
        R7 = self.R7[item]
        R8 = self.R8[item]
        R9 = self.R9[item]
        T3 = self.T3[item]


        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # randomly flip images in an image pair
        '''if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])'''

        return {'img1': img1,
                'img2': img2,
                'R1': R1,
                'R2' : R2,
                'R3' : R3,
                'T1' : T1,
                'R4' : R4,
                'R5' : R5,
                'R6' : R6,
                'T2' : T2,
                'R7' : R7,
                'R8' : R8,
                'R9' : R9,
                'T3' : T3}

    def __len__(self):
        return len(self.fnames1)

############################ DTU DATASET TRAIN ##################################
class DTUDatasetClassify(object):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        #self.scenes_dict = defaultdict(str)
        #for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
        #    self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.gt_labels = self._read_pairs_txt()

    def _read_pairs_txt(self):
        #fnames1, fnames2, t_gt, q_gt = [], [], [], []
        fnames1, fnames2, gt_labels = [], [], []

        #data_params = self.cfg.data_params

        #pairs_txt = data_params.train_pairs_fname if self.split == 'train' else data_params.val_pairs_fname
        if self.split == 'train':
            pairs_txt = './datasets/lists/dtu/train_DTU_classify.txt'
        else:
            pairs_txt = './datasets/lists/dtu/test_DTU_classify.txt'
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                #scene_id = int(chunks[2])
                #print((self.root_dir + chunks[0]))
                fnames1.append((self.root_dir + chunks[0]))
                fnames2.append((self.root_dir + chunks[1]))
                gt_labels.append(float(chunks[2]))

        return fnames1, fnames2, gt_labels

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        #t_gt = self.t_gt[item]
        #q_gt = self.q_gt[item]
        gt_labels = self.gt_labels[item]
        fnames1 = self.fnames1[item]
        fnames2 = self.fnames2[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # randomly flip images in an image pair
        '''if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])'''

        return {'img1': img1,
                'img2': img2,
                'gt_labels': gt_labels,
                'fnames1' : fnames1,
                'fnames2' : fnames2}

    def __len__(self):
        return len(self.fnames1)

############################ DTU DATASET TEST SET ##################################
class DTUTestDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.transforms = transforms
        #self.scenes_dict = defaultdict(str)
        #for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
        #    self.scenes_dict[i] = scene

        self.fnames1, self.fnames2 = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2 = [], []

        pairs_txt = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                #scene_id1 = int(chunks[2])
                #scene_id2 = int(chunks[3])
                fnames1.append(osp.join(img_dir,  chunks[0][1:]))
                fnames2.append(osp.join(img_dir,  chunks[1][1:]))

        return fnames1, fnames2

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return {'img1': img1,
                'img2': img2,
                }

    def __len__(self):
        return len(self.fnames1)

############################## AALTO DATASET ###########################################

class AALTORelPoseDataset(object):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['office', 'meeting', 'kitchen1', 'conference', 'kitchen2']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2, self.t_gt, self.q_gt, self.kps = self._read_pairs_txt()



    def _read_pairs_txt(self):
        fnames1, fnames2, t_gt, q_gt, kps = [], [], [], [], []

        if(self.split == 'train'):
            with open(self.root_dir + 'db_aalto_all_train.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')
                    scene_id = int(chunks[2])

                    f1 = chunks[0][14:-4]
                    f2 = chunks[1][14:-4]

                    kp_filename = self.root_dir + '/KPS/KPS/' + self.scenes_dict[scene_id] + '/' + chunks[0][
                                                                                                   1:8] + 'points_' + f1 + '_' + f2 + '.txt'
                    if os.path.isfile(kp_filename):
                        this_kps = self.read_kps(kp_filename)
                        if(this_kps.shape[0] > 4):
                            temp_kps = this_kps[np.arange(0, this_kps.shape[0], int(np.floor((this_kps.shape[0]) / 4))), :]
                            if (temp_kps.shape[0] > 4):
                                kps += [(temp_kps[:4, :])]  # --> all keypoints used for calculating the conv layer based loss
                            else:
                                kps += [(temp_kps)]  # --> all keypoints used for calculating the conv layer based loss

                            fnames1.append(osp.join(self.root_dir, self.scenes_dict[scene_id], chunks[0][1:]))
                            fnames2.append(osp.join(self.root_dir, self.scenes_dict[scene_id], chunks[1][1:]))

                            t_gt.append(torch.FloatTensor([float(chunks[3]), float(chunks[4]), float(chunks[5])]))
                            q_gt.append(torch.FloatTensor([float(chunks[6]),
                                                           float(chunks[7]),
                                                           float(chunks[8]),
                                                           float(chunks[9])]))



        if (self.split == 'val'):
            with open(self.root_dir + 'db_aalto_all_valid.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')
                    scene_id = int(chunks[2])
                    fnames1.append(osp.join(self.root_dir, self.scenes_dict[scene_id], chunks[0][1:]))
                    fnames2.append(osp.join(self.root_dir, self.scenes_dict[scene_id], chunks[1][1:]))

                    t_gt.append(torch.FloatTensor([float(chunks[3]), float(chunks[4]), float(chunks[5])]))
                    q_gt.append(torch.FloatTensor([float(chunks[6]),
                                                   float(chunks[7]),
                                                   float(chunks[8]),
                                                   float(chunks[9])]))

        return fnames1, fnames2, t_gt, q_gt, kps

    def read_kps(self, kp_filename):
        with open(kp_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # each line contains pair of points, i.e., ref_point[y,x] and src_point[y,x]
        all_pts = np.fromstring(' '.join(lines[:]), dtype=np.float32, sep=' ')
        kps = all_pts.reshape((int(len(all_pts)/4), 4))
        return kps # no.ofpoints x 4 array

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]
        kps_ = self.kps[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        # randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])

        return {'fname1' : self.fnames1[item],
                'fname2' : self.fnames2[item],
                'img1': img1,
                'img2': img2,
                't_gt': t_gt,
                'q_gt': q_gt,
                'kps' : kps_}

    def __len__(self):
        return len(self.fnames1)


class SevenScenesTestDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.experiment_cfg = experiment_cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['office', 'meeting', 'kitchen1', 'conference', 'kitchen2']):
            self.scenes_dict[i] = scene

        self.fnames1, self.fnames2 = self._read_pairs_txt()

    def _read_pairs_txt(self):
        fnames1, fnames2 = [], []

        pairs_txt = self.experiment_cfg.paths.test_pairs_fname
        img_dir = self.experiment_cfg.paths.img_path
        with open(pairs_txt, 'r') as f:
            for line in f:
                chunks = line.rstrip().split(' ')
                scene_id1 = int(chunks[2])
                scene_id2 = int(chunks[3])
                fnames1.append(osp.join(img_dir, self.scenes_dict[scene_id2], chunks[1][1:]))
                fnames2.append(osp.join(img_dir, self.scenes_dict[scene_id1], chunks[0][1:]))

        return fnames1, fnames2

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        return {'img1': img1,
                'img2': img2,
                }

    def __len__(self):
        return len(self.fnames1)


############################## BlendedMVS DATASET ###########################################

class BlendedMVSDataset(object):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        for i, scene in enumerate(['office', 'meeting', 'kitchen1', 'conference', 'kitchen2']):
            self.scenes_dict[i] = scene

        if(1):
            self.fnames1, self.fnames2, self.t_gt, self.q_gt, self.kps = self._read_pairs_txt_KPS()
        else:
            self.fnames1, self.fnames2, self.t_gt, self.q_gt = self._read_pairs_txt()



    def _read_pairs_txt(self):
        fnames1, fnames2, t_gt, q_gt = [], [], [], []

        if(self.split == 'train'):
            with open(self.root_dir + 'train_BlendedMVS.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')

                    scan_folder = chunks[0][1:10]

                    #f1 = chunks[0][11:-4] #filename
                    #f2 = chunks[1][11:-4] #filename

                    fnames1.append(osp.join(self.root_dir, scan_folder, 'blended_images' ,chunks[0][11:]))
                    fnames2.append(osp.join(self.root_dir, scan_folder, 'blended_images' ,chunks[1][11:]))

                    t_gt.append(torch.FloatTensor([float(chunks[6]), float(chunks[7]), float(chunks[8])]))
                    q_gt.append(torch.FloatTensor([float(chunks[2]),
                                                   float(chunks[3]),
                                                   float(chunks[4]),
                                                   float(chunks[5])]))

        if (self.split == 'test'):
            with open(self.root_dir + 'test_BlendedMVS.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')

                    scan_folder = chunks[0][1:10]

                    #f1 = chunks[0][11:-4]  # filename
                    #f2 = chunks[1][11:-4]  # filename

                    fnames1.append(osp.join(self.root_dir, scan_folder, 'blended_images', chunks[0][11:]))
                    fnames2.append(osp.join(self.root_dir, scan_folder, 'blended_images', chunks[1][11:]))

                    t_gt.append(torch.FloatTensor([float(chunks[6]), float(chunks[7]), float(chunks[8])]))
                    q_gt.append(torch.FloatTensor([float(chunks[2]),
                                                   float(chunks[3]),
                                                   float(chunks[4]),
                                                   float(chunks[5])]))

        return fnames1, fnames2, t_gt, q_gt

    def _read_pairs_txt_KPS(self):
        fnames1, fnames2, t_gt, q_gt, kps = [], [], [], [], []

        if(self.split == 'train'):
            with open(self.root_dir + 'train_BlendedMVS.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')

                    scan_folder = chunks[0][1:10]

                    f1 = chunks[0][11:-4] #filename
                    f2 = chunks[1][11:-4] #filename

                    kp_filename = self.root_dir + '/KPS/' + scan_folder + '/' + 'points_' + f1 + '_' + f2 + '.txt'
                    if os.path.isfile(kp_filename):
                        this_kps = self.read_kps(kp_filename)
                        if(this_kps.shape[0] > 4):
                            temp_kps = this_kps[np.arange(0, this_kps.shape[0], int(np.floor((this_kps.shape[0]) / 4))), :]
                            if (temp_kps.shape[0] > 4):
                                kps += [(temp_kps[:4, :])]  # --> all keypoints used for calculating the conv layer based loss
                            else:
                                kps += [(temp_kps)]  # --> all keypoints used for calculating the conv layer based loss

                            fnames1.append(osp.join(self.root_dir, scan_folder, 'blended_images' ,chunks[0][11:]))
                            fnames2.append(osp.join(self.root_dir, scan_folder, 'blended_images' ,chunks[1][11:]))

                            t_gt.append(torch.FloatTensor([float(chunks[6]), float(chunks[7]), float(chunks[8])]))
                            q_gt.append(torch.FloatTensor([float(chunks[2]),
                                                           float(chunks[3]),
                                                           float(chunks[4]),
                                                           float(chunks[5])]))

        if (self.split == 'test'):
            with open(self.root_dir + 'test_BlendedMVS.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split(' ')

                    scan_folder = chunks[0][1:10]

                    f1 = chunks[0][11:-4]  # filename
                    f2 = chunks[1][11:-4]  # filename

                    kp_filename = self.root_dir + '/KPS/' + scan_folder + '/' + 'points_' + f1 + '_' + f2 + '.txt'
                    if os.path.isfile(kp_filename):
                        this_kps = self.read_kps(kp_filename)
                        if (this_kps.shape[0] > 4):
                            temp_kps = this_kps[np.arange(0, this_kps.shape[0], int(np.floor((this_kps.shape[0]) / 4))),
                                       :]
                            if (temp_kps.shape[0] > 4):
                                kps += [(temp_kps[:4,
                                         :])]  # --> all keypoints used for calculating the conv layer based loss
                            else:
                                kps += [(temp_kps)]  # --> all keypoints used for calculating the conv layer based loss

                            fnames1.append(osp.join(self.root_dir, scan_folder, 'blended_images', chunks[0][11:]))
                            fnames2.append(osp.join(self.root_dir, scan_folder, 'blended_images', chunks[1][11:]))

                            t_gt.append(torch.FloatTensor([float(chunks[6]), float(chunks[7]), float(chunks[8])]))
                            q_gt.append(torch.FloatTensor([float(chunks[2]),
                                                           float(chunks[3]),
                                                           float(chunks[4]),
                                                           float(chunks[5])]))

        return fnames1, fnames2, t_gt, q_gt, kps

    def read_kps(self, kp_filename):
        with open(kp_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # each line contains pair of points, i.e., ref_point[y,x] and src_point[y,x]
        all_pts = np.fromstring(' '.join(lines[:]), dtype=np.float32, sep=' ')
        kps = all_pts.reshape((int(len(all_pts)/4), 4))
        return kps # no.ofpoints x 4 array

    def __getitem__(self, item):
        img1 = Image.open(self.fnames1[item]).convert('RGB')
        img2 = Image.open(self.fnames2[item]).convert('RGB')
        t_gt = self.t_gt[item]
        q_gt = self.q_gt[item]
        if(1):
            kps_ = self.kps[item]

        if self.transforms:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)

        '''# randomly flip images in an image pair
        if random.uniform(0, 1) > 0.5:
            img1, img2 = img2, img1
            t_gt = -self.t_gt[item]
            q_gt = torch.FloatTensor([q_gt[0], -q_gt[1], -q_gt[2], -q_gt[3]])'''

        if(1):
            return {'fname1': self.fnames1[item],
                    'fname2': self.fnames2[item],
                    'img1': img1,
                    'img2': img2,
                    't_gt': t_gt,
                    'q_gt': q_gt,
                    'kps' : kps_}
        else:
            return {'fname1': self.fnames1[item],
                    'fname2': self.fnames2[item],
                    'img1': img1,
                    'img2': img2,
                    't_gt': t_gt,
                    'q_gt': q_gt}

    def __len__(self):
        return len(self.fnames1)
