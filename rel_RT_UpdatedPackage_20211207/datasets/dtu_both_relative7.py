from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R_
import copy

class DTUDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, n_depths=256, interval_scale=1.06, img_size=256, img_crop=224, intrinsic_file_path='intrinsics.txt'):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'val', 'test'], \
            'split must be either "train" or "val"!'
        self.build_metas()
        self.n_views = n_views
        self.n_depths = n_depths
        self.interval_scale = interval_scale
        self.img_size = img_size
        self.img_crop = img_crop
        self.build_proj_mats()
        self.define_transforms()
        self.depth_min = 0
        self.depth_interval = 0
        self.read_intrinsics_file(intrinsic_file_path)

    def build_metas(self):
        self.metas = []
        with open(f'datasets/lists/dtu/{self.split}.txt') as f:
            scans = [line.rstrip() for line in f.readlines()]

        pair_file = "Cameras/pair.txt"
        for scan in scans:
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if(self.split == 'train'):
                        # light conditions 0-6
                        for light_idx in range(7):
                            self.metas += [(scan, light_idx, ref_view, src_views)]

                    if(self.split == 'test'):
                        # light conditions 0 'only use lighting condition 0
                        for light_idx in range(1):
                            self.metas += [(scan, light_idx, ref_view, src_views)]

    def build_proj_mats(self):
        proj_mats = []

        for vid in range(49): # total 49 view ids
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsics, extrinsics, depth_min, depth_interval = \
                self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            #proj_mat = copy.deepcopy(extrinsics)
            extrinsics_ = copy.deepcopy(extrinsics)
            rotat = extrinsics[:3,:3]
            trans = extrinsics[:3, 3]
            #print("extrinsics before indx :{} = {} ".format(vid,extrinsics_))
            #proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
            #print("extrinsics after proj indx :{} = {} ".format(vid, extrinsics_))
            #r = R_.from_matrix(extrinsics_[:3,:3])
            #qvec = r.as_quat()
            #x = float(extrinsics[0,3])
            #y = float(extrinsics[1,3])
            #z = float(extrinsics[2,3])
            #print("extrinsics after quat indx :{} = {}".format(vid, extrinsics))
            #print("x={}, y={},z={}".format(x,y,z))
            #q1 = float(qvec[3])
            #q2 = float(qvec[0])
            #q3 = float(qvec[1])
            #q4 = float(qvec[2])
            #quat = np.array([q1 ,q2 ,q3 ,q4, x, y, z])
            #print("quat vals index:{} = {}".format(vid, quat))
            proj_mats += [(rotat, trans, depth_min, depth_interval)]

        self.proj_mats = proj_mats

    def read_intrinsics_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # intrinsics: line [1-3), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[1:4]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[5].split()[0])
        self.depth_min = depth_min
        depth_interval = float(lines[5].split()[1]) * self.interval_scale
        self.depth_interval = depth_interval

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_depth(self, filename):
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def define_transforms(self):
        if self.split == 'train':
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
            self.transform_RT = T.Compose([T.Resize(self.img_size),
                                        T.CenterCrop(self.img_crop),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])
        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
            self.transform_RT = T.Compose([T.Resize(self.img_size),
                                        T.CenterCrop(self.img_crop),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, light_idx, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs = []
        imgs_RT = []
        proj_mats_source = []
        proj_mats_ref = []
        quat_ = []
        rel_quat_ = []
        ref_imgs = [] #will contain the copies of ref imgs, to be used in batch fashion
        src_imgs = [] #will contain the source views
        R_temp = []
        T_temp = []
        filename_to_save_ref = []
        filename_to_save_src = []
        view_ids_ = []

        #print('view ids = {}'.format(view_ids))

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                f'Rectified/{scan}_train/rect_{vid+1:03d}_{light_idx}_r5000.png')
            mask_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}_train/depth_visual_{vid:04d}.png')
            depth_filename = os.path.join(self.root_dir,
                                f'Depths/{scan}_train/depth_map_{vid:04d}.pfm')

            img = Image.open(img_filename)
            img_cpy = img.copy()
            img = self.transform(img)
            img_cpy = self.transform_RT(img_cpy)

            imgs += [img]
            imgs_RT += [img_cpy]

            Rot, Tra, depth_min, depth_interval = self.proj_mats[vid]
            #print('filename to save i = {} : {}'.format(i, f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png'))

            if i == 0:  # reference view
                depth_values = torch.arange(self.depth_min,
                                            self.depth_interval*self.n_depths+self.depth_min,
                                            self.depth_interval,
                                            dtype=torch.float32)
                mask = Image.open(mask_filename)
                mask = torch.BoolTensor(np.array(mask))
                depth = torch.FloatTensor(self.read_depth(depth_filename))
                #filename_to_save = f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png'
                filename_to_save_ref += [f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png']
                #print('filename to save : ',filename_to_save)
                R_temp += [Rot]
                T_temp += [Tra]
                #proj_mats_source += [torch.inverse(proj_mat)]
                for j in range(self.n_views-1):
                    ref_imgs += [img_cpy] # as many copies of ref images as the number of source images
                #quat_ += [torch.FloatTensor(quat)]
            else:
                filename_to_save_src += [f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png']
                #proj_mats_ref += [proj_mat]
                src_imgs += [img_cpy]
                #R_temp_ = np.linalg.inv(R_temp[0]) @ Rot
                #T_temp_ = np.linalg.inv(R_temp[0]) @ (Tra - T_temp[0])
                R_temp_ = Rot @ np.linalg.inv(R_temp[0])
                T_temp_ = (Tra - T_temp[0]) @ np.linalg.inv(R_temp[0])
                # print("extrinsics after proj indx :{} = {} ".format(vid, extrinsics_))
                r = R_.from_matrix(R_temp_)
                qvec = r.as_quat()
                x = float(T_temp_[0])
                y = float(T_temp_[1])
                z = float(T_temp_[2])
                # print("extrinsics after quat indx :{} = {}".format(vid, extrinsics))
                #print("x={}, y={},z={}".format(x,y,z))
                q1 = float(qvec[3])
                q2 = float(qvec[0])
                q3 = float(qvec[1])
                q4 = float(qvec[2])
                rel_quat = np.array([q1 ,q2 ,q3 ,q4, x, y, z])
                #print("quat vals index:{} = {}".format(vid, rel_quat))
                rel_quat_ += [torch.FloatTensor(rel_quat)]
                view_ids_ += [vid]



        imgs = torch.stack(imgs)
        #imgs_RT = torch.stack(imgs_RT)
        #proj_mats = torch.stack(proj_mats)
        #quat_ = torch.stack(quat_)
        rel_quat_ = torch.stack(rel_quat_)
        src_imgs = torch.stack(src_imgs)
        ref_imgs = torch.stack(ref_imgs)

        #return imgs, proj_mats, depth, depth_values, mask, imgs_RT
        #return imgs, depth, depth_values, mask, imgs_RT, filename_to_save, proj_mats, quat_

        if (self.split == 'train'):
            return ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, depth, mask  # when training relative RT net only

        if (self.split == 'test'):
            return  ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, imgs, depth, depth_values, mask, view_ids_ # when training and testing relative RTNet with MVSNET

        '''if (self.split == 'test'):
            return  ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src # when testing relative RTNet only (without mvsnet)'''