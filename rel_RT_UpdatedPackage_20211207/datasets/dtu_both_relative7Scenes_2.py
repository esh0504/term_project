from torch.utils.data import Dataset
from .utils import read_pfm
import os
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R_
import copy

class SevenScenes(Dataset):
    def __init__(self, root_dir, split, n_views=3, n_depths=256, interval_scale=1.06, img_size=256, img_crop=224):
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
        self.define_transforms()
        self.depth_min = 0
        self.depth_interval = 0
        #self.read_intrinsics_file(intrinsic_file_path)
        self.view_points = 1000
        self.scene = ''

    def build_metas(self):
        self.metas = []
        self.scene = self.root_dir.split('/')[-2]
        with open(f'datasets/lists/7Scenes/{self.split}_{self.scene}.txt') as f:
            scans = [line.rstrip() for line in f.readlines()]

        for scan in scans:
            #pair_file = f'{scan}/pair.txt'
            #pair_file = f'{scan}/pair_500.txt'
            pair_file = f'{scan}/pair_top_5.txt'
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                self.view_points = num_viewpoint
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    #print('Scan : {} , ref_view : {}, src_views : {}'.format(scan, ref_view, src_views))
                    # self.metas += [(scan, ref_view, src_views)]
                    if (self.split == 'train'):
                        for y in range(len(src_views)):
                            self.metas += [(scan, ref_view, src_views[y])] # use only a pair of 2 (1 ref and 1 src image), use all the combinations

                    if (self.split == 'test'):
                        if (len(src_views) > 5):
                            top_5 = 5
                        else:
                            top_5 = len(src_views)
                        for y in range(top_5):#(len(src_views)):
                            self.metas += [(scan, ref_view, src_views[y])] # use only a pair of 2 (1 ref and 1 src image), use all the combinations


    '''def build_proj_mats(self):
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
            #proj_mats += [(rotat, trans, intrinsics, depth_min, depth_interval)]
            proj_mats += [(extrinsics, intrinsics, depth_min, depth_interval)]

        self.proj_mats = proj_mats'''

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

    def read_cam_file(self, filename, vid):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))

        #print('Extrinsics : {}'.format(extrinsics))

        return extrinsics

    def read_depth(self, filename):
        #return np.array(read_pfm(filename)[0], dtype=np.float32)
        return np.array(Image.open(filename), dtype=np.float32)

    def read_kps(self, kp_filename):
        with open(kp_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # each line contains pair of points, i.e., ref_point[y,x] and src_point[y,x]
        all_pts = np.fromstring(' '.join(lines[:]), dtype=np.float32, sep=' ')
        kps = all_pts.reshape((int(len(all_pts)/4), 4))
        return kps # no.ofpoints x 4 array

    def define_transforms(self):
        if self.split == 'train':
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
            '''self.transform_RT = T.Compose([T.Resize(self.img_size),
                                        T.CenterCrop(self.img_crop),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])'''
            self.transform_RT = T.Compose([T.Resize((224,224)),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                           ])

        else:
            self.transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                       ])
            '''self.transform_RT = T.Compose([T.Resize(self.img_size),
                                        T.CenterCrop(self.img_crop),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])'''
            self.transform_RT = T.Compose([T.Resize((224,224)),
                                           T.ToTensor(),
                                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                                           ])


    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        scan, ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        #view_ids = [ref_view] + src_views[:self.n_views-1]
        view_ids = [ref_view] + [src_views]
        #print('view_ids : {}'.format(view_ids))

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
        Extr_temp = []
        Extr_temp2 = []
        filename_to_save_ref = []
        filename_to_save_src = []
        view_ids_ = []
        kps_ = []
        ref_id_ = 0
        self.scene = self.root_dir.split('/')[-2]


        #print('view ids = {}'.format(view_ids))

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,f'{scan}/frame-{vid:06d}.color.png')
            #print("img filename : {}".format(img_filename))
            depth_filename = os.path.join(self.root_dir,f'{scan}/frame-{vid:06d}.depth.png')
            #print("depth filename : {}".format(depth_filename))
            img = Image.open(img_filename)
            img_cpy = img.copy()

            #Test
            #fnt = ImageFont.truetype("arial.ttf", 100)
            #d = ImageDraw.Draw(img_cpy)
            #d.multiline_text((10, 10), f'rect_{vid+1:03d}_{light_idx}_r5000.png', font=fnt, fill=(255, 255, 0))
            #test

            img = self.transform(img)
            img_cpy = self.transform_RT(img_cpy)

            imgs += [img]
            imgs_RT += [img_cpy]

            proj_mat_filename = os.path.join(self.root_dir,f'{scan}/frame-{vid:06d}.pose.txt')
            #print("proj_mat_filename : {}".format(proj_mat_filename))
            Extrinsic = self.read_cam_file(proj_mat_filename, vid)
            #Extrinsic, Intrinsic, depth_min, depth_interval = self.proj_mats[vid]
            #print('filename to save i = {} : {}'.format(i, f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png'))

            if i == 0:  # reference view
                ref_id_ = vid
                depth = torch.FloatTensor(self.read_depth(depth_filename))
                #filename_to_save = f'{scan}_train_rect_{vid+1:03d}_{light_idx}_r5000.png'
                filename_to_save_ref += [f'{scan}/frame-{vid:06d}.color.png']
                #print('filename to save : ',filename_to_save)
                #R_temp += [Rot]
                #T_temp += [Tra]
                Extr_temp += [Extrinsic]
                #proj_mats_source += [torch.inverse(proj_mat)]
                for j in range(self.n_views-1):
                    ref_imgs += [img_cpy] # as many copies of ref images as the number of source images
                #quat_ += [torch.FloatTensor(quat)]
            else:
                filename_to_save_src += [f'{scan}/frame-{vid:06d}.color.png']
                #proj_mats_ref += [proj_mat]
                src_imgs += [img_cpy]
                #R_temp_ = np.linalg.inv(R_temp[0]) @ Rot
                #T_temp_ = np.linalg.inv(R_temp[0]) @ (Tra - T_temp[0])
                Extr_temp_ = Extrinsic @ np.linalg.inv(Extr_temp[0])
                extrinsics_ = copy.deepcopy(Extr_temp_)
                #Extr_temp2 += [copy.deepcopy(Extr_temp_)]
                #R_temp_ = Intrinsic @ Extr_temp_[:3, :3]
                #T_temp_ = Intrinsic @ extrinsics_[:3, 3]
                T_temp_ = Extr_temp_[:3, 3]
                #T_temp_ = (Tra - T_temp[0]) @ np.linalg.inv(R_temp[0])
                # print("extrinsics after proj indx :{} = {} ".format(vid, extrinsics_))
                # print("vid : {}, Intr = {}, rel_P2 = {}, R_temp_ = {} ".format(vid, Intrinsic, extrinsics_, R_temp_))
                r = R_.from_matrix(extrinsics_[:3, :3])
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
                kp_filename = '../7Scenes/KPS/KPS/' + f'{self.scene}/{scan}/points_{ref_id_:06d}_{vid:06d}.txt'
                this_kps = self.read_kps(kp_filename)
                #kps_ += [torch.FloatTensor(this_kps[np.random.choice(this_kps.shape[0],4),:])] # --> 4 is the number of keypoints used for calculating the reporjection based loss
                #kps_ += [torch.FloatTensor(this_kps[np.random.choice(this_kps.shape[0], 10),:])]  # --> 10 is the number of keypoints used for calculating the conv layer based loss
                temp_kps = this_kps[np.arange(0, this_kps.shape[0], int(np.floor((this_kps.shape[0]) / 20))), :]
                if(temp_kps.shape[0] > 20):
                    kps_ += [torch.FloatTensor(temp_kps[:20,:])]  # --> all keypoints used for calculating the conv layer based loss
                else:
                    kps_ += [torch.FloatTensor(temp_kps)]  # --> all keypoints used for calculating the conv layer based loss
                #print('kps_ shape here : ',kps_[i-1].shape)




        imgs = torch.stack(imgs)
        #imgs_RT = torch.stack(imgs_RT)
        #proj_mats = torch.stack(proj_mats)
        #quat_ = torch.stack(quat_)
        rel_quat_ = torch.stack(rel_quat_)
        src_imgs = torch.stack(src_imgs)
        ref_imgs = torch.stack(ref_imgs)
        #print("size of kps :", kps_[0].shape)
        kps_ = torch.stack(kps_)


        #return imgs, proj_mats, depth, depth_values, mask, imgs_RT
        #return imgs, depth, depth_values, mask, imgs_RT, filename_to_save, proj_mats, quat_

        if (self.split == 'train'):
            #return ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, Extr_temp2, depth, mask  # when training relative RT net only
            return ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, depth, kps_  # when training relative RT net only

        '''if (self.split == 'test'):
            return  ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src, imgs, depth, depth_values, mask, view_ids_ # when training and testing relative RTNet with MVSNET'''

        if (self.split == 'test'):
            return  ref_imgs, src_imgs, rel_quat_, filename_to_save_ref, filename_to_save_src # when testing relative RTNet only (without mvsnet)