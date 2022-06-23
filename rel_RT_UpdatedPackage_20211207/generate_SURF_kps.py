# 2D 포인트에 8방향 체크 후 임계값을 못넘기면 제거하기

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_
import re
import math
from matplotlib import pyplot as plt
boundary_away = 2
WANT_SIFT = 0
WANT_SURF = 1

foldernames = ['scan2','scan6','scan7','scan8','scan14','scan16','scan18','scan19','scan20','scan22','scan30','scan31','scan36','scan39','scan41','scan42','scan44','scan45','scan46','scan47','scan50','scan51','scan52','scan53','scan55','scan57','scan58','scan60','scan61','scan63','scan64','scan65','scan68','scan69','scan70','scan71','scan72','scan74','scan76','scan83','scan84','scan85','scan87','scan88','scan89','scan90','scan91','scan92','scan93','scan94','scan95','scan96','scan97','scan98','scan99','scan100','scan101','scan102','scan103','scan104','scan105','scan107','scan108','scan109','scan111','scan112','scan113','scan115','scan116','scan119','scan120','scan121','scan122','scan123','scan124','scan125','scan126','scan127','scan128','scan1',\
'scan4','scan9','scan10','scan11','scan12','scan13','scan15','scan23','scan24','scan29','scan32','scan33','scan34','scan48','scan49','scan62','scan75','scan77','scan110','scan114','scan118','scan3','scan5','scan17','scan21','scan28','scan35','scan37','scan38','scan40','scan43','scan56','scan59','scan66','scan67','scan82','scan86','scan106','scan117']

def center_crop(img, dim):
	#Returns center cropped image
	#Arg's:Image Scaling
	#img: image to be center cropped
	#dim: dimensions (width, height) to be cropped from center

    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    return intrinsics, extrinsics

def build_proj_mats(root_dir):
    proj_mats = []

    for vid in range(49):  # total 49 view ids
        proj_mat_filename = os.path.join(root_dir,
                                         f'Cameras/train/{vid:08d}_cam.txt')
        intrinsics, extrinsics = \
            read_cam_file(proj_mat_filename)

        # multiply intrinsics and extrinsics to get projection matrix
        # proj_mat = copy.deepcopy(extrinsics)
        proj_mats += [(extrinsics, intrinsics)]

    return proj_mats


def read_depth(filename):
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def inverse_project2(intrinsics, extrinsics, depths, kps_src, im1):

    proj_mat = extrinsics
    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

    #print("proj_mat={} ".format(proj_mat))
    kps_src = kps_src.astype(np.int32)

    pt2_2D = []

    #img1_4 = np.zeros((56,56))
    img1_4 = np.zeros(depths.shape)
    #overlapRegionImg1_4 = np.zeros((56,56))
    overlapRegionImg1_4 = np.zeros(depths.shape)
    out_of_img = np.zeros(depths.shape)

    for j in range(len(kps_src)):
        arr = np.zeros((3,1))
        arr[0] = depths[kps_src[j,1],kps_src[j,0]] * kps_src[j, 0]
        arr[1] = depths[kps_src[j,1],kps_src[j,0]] * kps_src[j, 1]
        arr[2] = depths[kps_src[j,1],kps_src[j,0]]
        #print('arr = {}'.format(arr))
        #pt1_3D = (np.linalg.inv(intrinsics) @ np.array[[depths[kps_src[j,0],kps_src[j,1]] * kps_src[j, 0]], [depths[kps_src[j,0],kps_src[j,1]] * kps_src[j, 1]], [depths[kps_src[j,0],kps_src[j,1]]]])
        pt1_3D = (np.linalg.inv(intrinsics) @ arr)
        #print('pt1_3D = {}'.format(pt1_3D))
        arr2 = np.ones((4, 1))
        arr2[:3] = pt1_3D
        #pt2 = proj_mat[:3,:4] @ pt1_3D
        pt2 = proj_mat[:3, :4] @ arr2
        '''if(depths[kps_src[j,1],kps_src[j,0]] > 0):
            print('Point1 2D={}'.format([kps_src[j,1],kps_src[j,0]]))
            print('Point2 2D={}'.format([pt2[1]/pt2[2], pt2[0]/pt2[2]]))'''
        #pt2_0 = (np.round((pt2[0] / pt2[2]) * (56 / 160))).astype(np.int32)
        #pt2_1 = (np.round((pt2[1] / pt2[2]) * (56 / 128))).astype(np.int32)
        pt2_0 = np.round(pt2[0] / pt2[2]).astype(np.int16)
        pt2_1 = np.round(pt2[1] / pt2[2]).astype(np.int16)
        pt2_2D.append([pt2_0, pt2_1])

        #print('type of pt2 = {}'.format(pt2_1, [pt2_0]))

        if (pt2_0 < img1_4.shape[1] and pt2_1 < img1_4.shape[0] and pt2_0 > 0 and pt2_1 > 0):
            img1_4[pt2_1, pt2_0] = im1[kps_src[j,1].astype(np.uint16),kps_src[j,0].astype(np.uint16)] # --> project pixels from Ref to Src
            overlapRegionImg1_4[pt2_1, pt2_0] = overlapRegionImg1_4[pt2_1, pt2_0] + 1;
        '''else:
            out_of_img[kps_src[j,1],kps_src[j,0]] = im1[kps_src[j,1],kps_src[j,0]]'''

    return pt2_2D, img1_4, overlapRegionImg1_4, out_of_img

root_dir = '../../../zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/'

all_proj_mats = build_proj_mats(root_dir)

dtu_dataset_folder = '../../../zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/Rectified/'
keypoints_folder = './esh_keypoints_with_SURF_ver1.1/'
img_folder = './esh_keypoints_with_SURF_ver1.1/projec_imgs/'

if not os.path.exists(keypoints_folder):
    os.mkdir(keypoints_folder)

if not os.path.exists(img_folder):
    os.mkdir(img_folder)

if WANT_SIFT:
    sift = cv2.xfeatures2d.SIFT_create() #initial sift detector
    keypoints_folder = './esh_keypoints_with_SIFT_ver1.1/'
    img_folder = './esh_keypoints_with_SIFT_ver1.1/projec_imgs/'

if WANT_SURF:
    surf = cv2.xfeatures2d.SURF_create()
    keypoints_folder = './esh_keypoints_with_SURF_ver1.1/'
    img_folder = './esh_keypoints_with_SURF_ver1.1/projec_imgs/'

MIN_MATCH_COUNT = 10

folder_names = os.listdir(dtu_dataset_folder) #get the foldernames in dtu dataset

size_of_dir = len(folder_names)

all_pts_ref = np.array([[19, 19]])
for y in np.arange(20, 110, 5):
    for x in np.arange(20, 120, 5):
        all_pts_ref = np.append(all_pts_ref, [[x, y]], axis=0)
        # print('x = {}, y = {}, all_pts_ref = {}'.format(x,y,all_pts_ref))
        # np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)

for file_count in range(len(foldernames)): # get the filenames in each folder of dtu dataset
    folder_name = foldernames[file_count] + '_train'
    if not os.path.exists(keypoints_folder + folder_name):
        os.mkdir(keypoints_folder + folder_name)

    if not os.path.exists(img_folder + folder_name):
        os.mkdir(img_folder + folder_name)

    #filenames = os.listdir(dtu_dataset_folder + folder_name + '/')

    for vid_ref in range(49): # for each file in dataset, find SIFT features and save in a txt file
        filename_to_ref =  f'rect_{vid_ref+1:03d}_5_r5000.png'
        #print('img filename = {}'.format(dtu_dataset_folder + folder_name + '/' + filename_to_ref))
        ref_im = cv2.imread((dtu_dataset_folder + folder_name + '/' + filename_to_ref),cv2.IMREAD_GRAYSCALE)
        ref_im = cv2.resize(ref_im, (160, 128))
        if WANT_SIFT:
            ref_kp, ref_des = sift.detectAndCompute(ref_im,None) 
        if WANT_SURF:
            ref_kp, ref_des = surf.detectAndCompute(ref_im,None) 
            


        for vid_src in range(49):

            if(vid_ref != vid_src):
                filename_to_src = f'rect_{vid_src+1:03d}_5_r5000.png'
                src_im = cv2.imread((dtu_dataset_folder + folder_name + '/' + filename_to_src),cv2.IMREAD_GRAYSCALE)
                src_im = cv2.resize(src_im, (160, 128))
                if WANT_SIFT:
                    src_kp, src_des = sift.detectAndCompute(src_im,None) 
                if WANT_SURF:
                    src_kp, src_des = surf.detectAndCompute(src_im,None)
                    
                FLANN_INDEX_KDTREE = 0 
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
                search_params = dict(checks = 50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(ref_des,src_des,k=2)
                
                good = [] 
                for m,n in matches: 
                    if m.distance < 0.9*n.distance: 
                        good.append(m)
                

                if len(good)>MIN_MATCH_COUNT:
                    dst_pts = np.float32([ ref_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2) 
                    src_pts = np.float32([ src_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2) 
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist() 
                    h,w = ref_im.shape 
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) 
                    try:
                        fp = open(keypoints_folder + folder_name + '/' + f'points_{vid_ref+1:02d}_{vid_src+1:02d}.txt', "w")
                        for j in range(len(src_kp)):
                            all_src_kp = np.int32([src_kp[j].pt[0],src_kp[j].pt[1],1])
                            all_ref_kp = np.int32(np.dot(M, all_src_kp))
                            if all_ref_kp[0] < 0 or all_ref_kp[1] < 0 or all_ref_kp[0] > ref_im.shape[0] or all_ref_kp[1] > ref_im.shape[1]:
                                continue

                        
                            fp.write(str(all_ref_kp[0]) + ' ' + str(all_ref_kp[1]) + ' ' + str(all_src_kp[0]) + ' ' + str(all_src_kp[1])+'\n')
                        fp.close()
                    except:
                        continue
                    
                else: 
                    #print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)) 
                    matchesMask = None

                
                
                
                

                
               
    print('Files done : {} of {} \n'.format(file_count, size_of_dir))



