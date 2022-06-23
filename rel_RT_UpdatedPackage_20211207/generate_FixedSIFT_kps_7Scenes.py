import os
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R_
import re
import math

boundary_away = 2
WANT_SIFT = 1

foldernames = ['seq-01','seq-02','seq-04','seq-06']

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


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))

    # print('Extrinsics : {}'.format(extrinsics))

    return extrinsics

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

root_dir = '../7Scenes/chess/chess/'

dtu_dataset_folder = '../7Scenes/chess/chess/'
keypoints_folder = '../7Scenes/keypoints/'

if not os.path.exists(keypoints_folder):
    os.mkdir(keypoints_folder)

if WANT_SIFT:
    sift = cv.xfeatures2d.SIFT_create() #initial sift detector



folder_names = os.listdir(dtu_dataset_folder) #get the foldernames in dtu dataset

size_of_dir = len(folder_names)

Intrinsic = np.array([320., 0., 585.], [0., 240., 585.] [0., 0. ,1.], dtype=np.float32) # from the website https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

all_pts_ref = np.array([[19, 19]])
for y in np.arange(20, 110, 5):
    for x in np.arange(20, 120, 5):
        all_pts_ref = np.append(all_pts_ref, [[x, y]], axis=0)
        # print('x = {}, y = {}, all_pts_ref = {}'.format(x,y,all_pts_ref))
        # np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)

for file_count in range(len(foldernames)): # get the filenames in each folder of dtu dataset
    folder_name = foldernames[file_count]
    if not os.path.exists(keypoints_folder + folder_name):
        os.mkdir(keypoints_folder + folder_name)

    filenames = os.listdir(dtu_dataset_folder + folder_name + '/')

    for vid_ref in range(len(filenames)): # for each file in dataset, find SIFT features and save in a txt file
        filename_to_ref =  f'frame-{vid_ref:06d}.color.png'
        #print('img filename = {}'.format(dtu_dataset_folder + folder_name + '/' + filename_to_ref))
        im = cv.imread((dtu_dataset_folder + folder_name + '/' + filename_to_ref),cv.IMREAD_GRAYSCALE)
        #im = cv.resize(im, (160, 128))

        #print('im shp = {}'.format(im.shape))
        depth_filename = dtu_dataset_folder + folder_name + '/' + f'frame-{vid_ref:06d}.depth.png'
        depth = cv.imread((depth_filename), cv.IMREAD_GRAYSCALE)
        proj_mat_filename = dtu_dataset_folder + folder_name + '/' + f'frame-{vid_ref:06d}.pose.txt'
        # print("proj_mat_filename : {}".format(proj_mat_filename))
        Extrinsic = read_cam_file(proj_mat_filename)
        Extr_temp = Extrinsic
        print('P1 Projection matrix: {}'.format(Extr_temp))

        #print('filename_to_ref = {}'.format(f'{folder_name}_rect_{vid_ref+1:03d}_5_r5000.png'))

        for vid_src in range(len(filenames)):

            if(vid_ref != vid_src):
                filename_to_save_src = f'frame-{vid_src:06d}.color.png'
                #print('filename_to_save_src = {}'.format(filename_to_save_src))
                proj_mat_filename = dtu_dataset_folder + folder_name + '/' + f'frame-{vid_src:06d}.pose.txt'
                Extrinsic = read_cam_file(proj_mat_filename)

                print('P2 Projection matrix: {}'.format(Extrinsic))
                #proj_mats_ref += [proj_mat]
                Rel_Extr = Extrinsic @ np.linalg.inv(Extr_temp)

                #print('RElative Projection matrix: {}'.format(Rel_Extr))

                all_points_on_Src, img1_4, CountImg1_4, out_of_img = inverse_project2(Intrinsic, Rel_Extr, depth, all_pts_ref, im)

                #all_pts_ref[:, 0] = np.round(all_pts_ref[:, 0] * (56 / 160))  # --> rescale to 56x56 which is the output feature size
                #all_pts_ref[:, 1] = np.round(all_pts_ref[:, 1] * (56 / 128))  # --> rescale to 56x56 which is the output feature size

                all_points_on_Src = np.array(all_points_on_Src)

                '''all_points_on_Src[:, 0] = (
                        all_points_on_Src[:, 0] * (56 / 160))  # --> rescale to 56x56 which is the output feature size
                all_points_on_Src[:, 1] = (
                        all_points_on_Src[:, 1] * (56 / 128))  # --> rescale to 56x56 which is the output feature size'''

                all_pts_ref1 = all_pts_ref #.astype(np.int32)

                #all_points_on_Src = all_points_on_Src.astype(np.int32)

                h, w = np.array((56, 56))
                h, w = np.array((128, 160))

                #new_src_kps = []
                #new_ref_kps = []
                # print("mask size = {}".format(mask.shape))
                mask_ = mask.astype(np.uint8)
                # cv2.imwrite('mask.png',mask_)
                # print("mask size = {}, dtype = {}, ".format(mask_.shape, mask_.dtype))

                th = 2


                #mask_ = cv.resize(mask_, (56, 56))
                #img1_4_ = cv.resize(img1_4, (56, 56))
                #CountImg1_4_ = cv.resize(CountImg1_4, (56, 56))
                fp = open(keypoints_folder + folder_name + '/' + f'points_{vid_ref+1:02d}_{vid_src+1:02d}.txt', "w")
                for j in range(all_pts_ref1.shape[0] - 1):
                    #print('Ref = {},{} --> Src = {},{}'.format(all_pts_ref[j, 0], all_pts_ref[j, 1], all_points_on_Src[j, 0][0], all_points_on_Src[j, 1][0]))
                    if (all_pts_ref1[j][0] >= boundary_away and all_pts_ref1[j][0] < (w - boundary_away) and all_pts_ref1[j][
                        1] >= boundary_away and all_pts_ref1[j][1] < (h - boundary_away) and mask_[all_pts_ref1[j, 1], all_pts_ref1[j, 0]] > 0 and depth[all_pts_ref1[j, 1], all_pts_ref1[j, 0]] > 0\
                            and all_points_on_Src[j][0] >= 0 and all_points_on_Src[j][0] < (w- boundary_away) and all_points_on_Src[j][
                                1] >= 0 and all_points_on_Src[j][1] < (h - boundary_away)) \
                        and (abs(depth[all_pts_ref1[j, 1], all_pts_ref1[j, 0]-1] - depth[all_pts_ref1[j, 1], all_pts_ref1[j, 0]]) < 50) \
                            and (img1_4[all_points_on_Src[j][1], all_points_on_Src[j][0]] > 0 and CountImg1_4[all_points_on_Src[j][1], all_points_on_Src[j][0]] <th) \
                            and (CountImg1_4[all_points_on_Src[j][1]-1, all_points_on_Src[j][0]-1] <3 and CountImg1_4[all_points_on_Src[j][1]-1, all_points_on_Src[j][0]] <3 and CountImg1_4[all_points_on_Src[j][1]-1, all_points_on_Src[j][0]+1] <3 \
                            and CountImg1_4[all_points_on_Src[j][1] , all_points_on_Src[j][0] - 1] <3 and CountImg1_4[all_points_on_Src[j][1], all_points_on_Src[j][0]+1] <3 and CountImg1_4[all_points_on_Src[j][1]+1, all_points_on_Src[j][0]-1] <3 and CountImg1_4[all_points_on_Src[j][1]+1, all_points_on_Src[j][0]] <3 \
                            and CountImg1_4[all_points_on_Src[j][1] + 1, all_points_on_Src[j][0] + 1] <3) :  # only write points that are not close to boundary of image and lie on the object (mask) and do not occlude

                        #new_ref_kps.append(all_pts_ref[j])
                        #new_src_kps.append(all_points_on_Src[j])
                        fp.write(str(all_pts_ref1[j][0]) + ' ' + str(all_pts_ref1[j][1]) + ' ' + str((all_points_on_Src[j][0])[0]) + ' ' + str((all_points_on_Src[j][1])[0]) + '\n')

                fp.close()

                #cv.imwrite(img_folder + folder_name + '/' + f'img_{vid_ref+1:02d}_{vid_src+1:02d}.png', img1_4)
                #cv.imwrite(img_folder + folder_name + '/' + f'count_{vid_ref+1:02d}_{vid_src+1:02d}.png', CountImg1_4)
                #cv.imwrite('img_folder + folder_name + '/' + f'outImg_{vid_ref+1:02d}_{vid_src+1:02d}.png'', out_of_img)
                #quit()
        #quit()

    print('Files done : {} of {} \n'.format(file_count, size_of_dir))



