import io
import torch
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
#import re
import sys
from tensorflow.python.lib.io import file_io
import cv2
import re

epoch_no = '5'
outdir = 'combined_pytorch'
outdir = "./ckpts/esh_RT/exp1/"



'''
def RT_to_ProjectionMartices(intrinsics, qvec, T):

    proj_mats = []
    batch_size = qvec.shape[0] # --> tells how many images in the batch, and we consider that the batch size = 1

    qvec = F.normalize(qvec, p=2, dim=1)

    T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image

    for i in range(batch_size):
        R = torch.zeros([3,3])
        R[0,0] = 1 - 2 * qvec[i,2]*qvec[i,2] - 2 * qvec[i,3]*qvec[i,3]
        R[0,1] = 2 * qvec[i,1] * qvec[i,2] - 2 * qvec[i,0] * qvec[i,3]
        R[0,2] = 2 * qvec[i,3] * qvec[i,1] + 2 * qvec[i,0] * qvec[i,2]

        R[1,0] = 2 * qvec[i,1] * qvec[i,2] + 2 * qvec[i,0] * qvec[i,3]
        R[1,1] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,3]*qvec[i,3]
        R[1,2] = 2 * qvec[i,2] * qvec[i,3] - 2 * qvec[i,0] * qvec[i,1]

        R[2,0] = 2 * qvec[i,3] * qvec[i,1] - 2 * qvec[i,0] * qvec[i,2]
        R[2,1] = 2 * qvec[i,2] * qvec[i,3] + 2 * qvec[i,0] * qvec[i,1]
        R[2,2] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,2]*qvec[i,2]

        proj_mat = torch.eye(4)

        proj_mat[:3, :3] = R
        proj_mat[:3, 3] = T[:,i]
        proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]



        if i == 0:

            if (0):

                R_ = R.cpu().numpy().astype(np.float32)
                T_ = T.cpu().numpy().astype(np.float32)
                text_file = open('./' + outdir + '/epoch_' + epoch_no + '/' + filename_to_save[0][:-4] + '.txt', "w")
                text_file.write('extrinsic\n')
                text_file.write(
                    str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0,i]) + '\n')
                text_file.write(
                    str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1,i]) + '\n')
                text_file.write(
                    str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2,i]) + '\n')
                text_file.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '1' + '\n\n')
                text_file.write('intrinsic\n')
                text_file.write('361.54125' + ' ' + '0' + ' ' + '82.900625\n')
                text_file.write('0' + ' ' + '360.3975' + ' ' + '66.383875\n')
                text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
                text_file.write('425.0' + ' ' + '2.5')
                text_file.close()
            proj_mats += [torch.inverse(proj_mat)]

        else:
            proj_mats += [proj_mat]

    proj_mats = torch.stack(proj_mats)

    proj_mats = proj_mats.unsqueeze(0).cuda() # to make the dimensions like ([1, 3, 4, 4]) from ([3, 4, 4])

    return proj_mats
'''

def RT_to_ProjectionMartices4(intrinsics, qvec, T, filename_to_save_ref, view_ids, outdir, epoch_no, set, printing):
    
    proj_mats = []
    batch_size = qvec.shape[0] # --> tells how many images in the batch, and we consider that the batch size = 1

    #print("batch size: {}".format(batch_size))
    T = torch.transpose(T, 0, 1)
    proj_mat = torch.eye(4)
    proj_mats += [proj_mat] # add the ref proj matrix as identity for passing to MVSNET

    #qvec = F.normalize(qvec, p=2, dim=1)
    
    for i in range(batch_size):
        R = torch.zeros([3,3])
        
        R[0,0] = 1 - 2 * qvec[i,2]*qvec[i,2] - 2 * qvec[i,3]*qvec[i,3]
        R[0,1] = 2 * qvec[i,1] * qvec[i,2] - 2 * qvec[i,0] * qvec[i,3]
        R[0,2] = 2 * qvec[i,3] * qvec[i,1] + 2 * qvec[i,0] * qvec[i,2]

        R[1,0] = 2 * qvec[i,1] * qvec[i,2] + 2 * qvec[i,0] * qvec[i,3]
        R[1,1] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,3]*qvec[i,3]
        R[1,2] = 2 * qvec[i,2] * qvec[i,3] - 2 * qvec[i,0] * qvec[i,1]

        R[2,0] = 2 * qvec[i,3] * qvec[i,1] - 2 * qvec[i,0] * qvec[i,2]
        R[2,1] = 2 * qvec[i,2] * qvec[i,3] + 2 * qvec[i,0] * qvec[i,1]
        R[2,2] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,2]*qvec[i,2]
        proj_mat = torch.eye(4)

        proj_mat[:3, :3] = R
        proj_mat[:3, 3] = T[:,i]
        proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]
            


        #if i == 0:

        if (printing):

            R_ = R.cpu().numpy().astype(np.float32)
            T_ = T.cpu().numpy().astype(np.float32)

            text_file = open('./' + outdir + '/epoch_' + epoch_no + '/' + set + filename_to_save_ref[0][:-4] + '_' + f'{view_ids[i].item()+1:03d}' + '_cam.txt', "w")
            text_file.write('extrinsic\n')
            text_file.write(
                str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0,i]) + '\n')
            text_file.write(
                str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1,i]) + '\n')
            text_file.write(
                str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2,i]) + '\n')
            text_file.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '1' + '\n\n')
            text_file.write('intrinsic\n')
            text_file.write('361.54125' + ' ' + '0' + ' ' + '82.900625\n')
            text_file.write('0' + ' ' + '360.3975' + ' ' + '66.383875\n')
            text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
            text_file.write('425.0' + ' ' + '2.5')
            text_file.close()

        #proj_mats += [torch.inverse(proj_mat)]

        #else:
        #    proj_mats += [proj_mat]

        proj_mats += [proj_mat]

    proj_mats = torch.stack(proj_mats)

    proj_mats = proj_mats.unsqueeze(0).cuda() # to make the dimensions like ([1, 3, 4, 4]) from ([3, 4, 4])

    return proj_mats

# for printing cam files
def RT_to_ProjectionMartices3(intrinsics, qvec, T, filename_to_save_ref, view_ids, outdir, epoch_no, set, printing):

    proj_mats = []
    batch_size = qvec.shape[0] # --> tells how many images in the batch, and we consider that the batch size = 1

    #print("batch size: {}".format(batch_size))
    T = torch.transpose(T, 0, 1)
    proj_mat = torch.eye(4)
    proj_mats += [proj_mat] # add the ref proj matrix as identity for passing to MVSNET

    #qvec = F.normalize(qvec, p=2, dim=1)
    
    for i in range(batch_size):
        R = torch.zeros([3,3])
        
        R[0,0] = 1 - 2 * qvec[i,2]*qvec[i,2] - 2 * qvec[i,3]*qvec[i,3]
        R[0,1] = 2 * qvec[i,1] * qvec[i,2] - 2 * qvec[i,0] * qvec[i,3]
        R[0,2] = 2 * qvec[i,3] * qvec[i,1] + 2 * qvec[i,0] * qvec[i,2]

        R[1,0] = 2 * qvec[i,1] * qvec[i,2] + 2 * qvec[i,0] * qvec[i,3]
        R[1,1] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,3]*qvec[i,3]
        R[1,2] = 2 * qvec[i,2] * qvec[i,3] - 2 * qvec[i,0] * qvec[i,1]

        R[2,0] = 2 * qvec[i,3] * qvec[i,1] - 2 * qvec[i,0] * qvec[i,2]
        R[2,1] = 2 * qvec[i,2] * qvec[i,3] + 2 * qvec[i,0] * qvec[i,1]
        R[2,2] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,2]*qvec[i,2]
        proj_mat = torch.eye(4)

        proj_mat[:3, :3] = R
        proj_mat[:3, 3] = T[:,i]
        proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]
            


        #if i == 0:

        if (printing):

            R_ = R.cpu().numpy().astype(np.float32)
            T_ = T.cpu().numpy().astype(np.float32)
            
            text_file = open('./' + outdir + '/epoch_' + epoch_no + '/' + set + filename_to_save_ref[0][:-4] + '_' + f'{view_ids[i].item()+1:03d}' + '_cam.txt', "w")
            text_file.write('extrinsic\n')
            text_file.write(
                str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0,i]) + '\n')
            text_file.write(
                str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1,i]) + '\n')
            text_file.write(
                str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2,i]) + '\n')
            text_file.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '1' + '\n\n')
            text_file.write('intrinsic\n')
            text_file.write('2892.33' + ' ' + '0' + ' ' + '823.206\n')
            text_file.write('0' + ' ' + '2883.18' + ' ' + '619.07\n')
            text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
            text_file.write('425.0' + ' ' + '2.5')
            text_file.close()

        #proj_mats += [torch.inverse(proj_mat)]

        #else:
        #    proj_mats += [proj_mat]

        proj_mats += [proj_mat]

    proj_mats = torch.stack(proj_mats)

    proj_mats = proj_mats.unsqueeze(0).cuda() # to make the dimensions like ([1, 3, 4, 4]) from ([3, 4, 4])

    return proj_mats


# for printing cam files
def RT_to_ProjectionMartices2(intrinsics, qvec, T, filename_to_save, outdir, epoch_no, set, printing):

    proj_mats = []
    batch_size = qvec.shape[0] # --> tells how many images in the batch, and we consider that the batch size = 1

    #qvec = F.normalize(qvec, p=2, dim=1)

    T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image

    for i in range(batch_size):
        R = torch.zeros([3,3])
        R[0,0] = 1 - 2 * qvec[i,2]*qvec[i,2] - 2 * qvec[i,3]*qvec[i,3]
        R[0,1] = 2 * qvec[i,1] * qvec[i,2] - 2 * qvec[i,0] * qvec[i,3]
        R[0,2] = 2 * qvec[i,3] * qvec[i,1] + 2 * qvec[i,0] * qvec[i,2]

        R[1,0] = 2 * qvec[i,1] * qvec[i,2] + 2 * qvec[i,0] * qvec[i,3]
        R[1,1] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,3]*qvec[i,3]
        R[1,2] = 2 * qvec[i,2] * qvec[i,3] - 2 * qvec[i,0] * qvec[i,1]

        R[2,0] = 2 * qvec[i,3] * qvec[i,1] - 2 * qvec[i,0] * qvec[i,2]
        R[2,1] = 2 * qvec[i,2] * qvec[i,3] + 2 * qvec[i,0] * qvec[i,1]
        R[2,2] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,2]*qvec[i,2]

        proj_mat = torch.eye(4)

        proj_mat[:3, :3] = R
        proj_mat[:3, 3] = T[:,i]
        proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]

        #print("pos_out={} \n, ori_out={} \n".format(T[:,i],qvec[i,:]))



        if i == 0:

            if (printing):

                R_ = R.cpu().numpy().astype(np.float32)
                T_ = T.cpu().numpy().astype(np.float32)
                text_file = open('./' + outdir + '/epoch_' + epoch_no + '/' + set + filename_to_save[0][:-4] + '_cam.txt', "w")
                text_file.write('extrinsic\n')
                text_file.write(
                    str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0,i]) + '\n')
                text_file.write(
                    str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1,i]) + '\n')
                text_file.write(
                    str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2,i]) + '\n')
                text_file.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '1' + '\n\n')
                text_file.write('intrinsic\n')
                text_file.write('361.54125' + ' ' + '0' + ' ' + '82.900625\n')
                text_file.write('0' + ' ' + '360.3975' + ' ' + '66.383875\n')
                text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
                text_file.write('425.0' + ' ' + '2.5')
                text_file.close()
            proj_mats += [torch.inverse(proj_mat)]

        else:
            proj_mats += [proj_mat]

    proj_mats = torch.stack(proj_mats)

    proj_mats = proj_mats.unsqueeze(0).cuda() # to make the dimensions like ([1, 3, 4, 4]) from ([3, 4, 4])

    return proj_mats

# for printing cam files
def RT_to_ProjectionMartices(intrinsics, qvec, T, filename_to_save):

    proj_mats = []
    batch_size = qvec.shape[0] # --> tells how many images in the batch, and we consider that the batch size = 1

    qvec = F.normalize(qvec, p=2, dim=1)

    T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image

    for i in range(batch_size):
        R = torch.zeros([3,3])
        R[0,0] = 1 - 2 * qvec[i,2]*qvec[i,2] - 2 * qvec[i,3]*qvec[i,3]
        R[0,1] = 2 * qvec[i,1] * qvec[i,2] - 2 * qvec[i,0] * qvec[i,3]
        R[0,2] = 2 * qvec[i,3] * qvec[i,1] + 2 * qvec[i,0] * qvec[i,2]

        R[1,0] = 2 * qvec[i,1] * qvec[i,2] + 2 * qvec[i,0] * qvec[i,3]
        R[1,1] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,3]*qvec[i,3]
        R[1,2] = 2 * qvec[i,2] * qvec[i,3] - 2 * qvec[i,0] * qvec[i,1]

        R[2,0] = 2 * qvec[i,3] * qvec[i,1] - 2 * qvec[i,0] * qvec[i,2]
        R[2,1] = 2 * qvec[i,2] * qvec[i,3] + 2 * qvec[i,0] * qvec[i,1]
        R[2,2] = 1 - 2 * qvec[i,1]*qvec[i,1] - 2 * qvec[i,2]*qvec[i,2]

        proj_mat = torch.eye(4)

        proj_mat[:3, :3] = R
        proj_mat[:3, 3] = T[:,i]
        proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]

        print("pos_out={} \n, ori_out={} \n".format(T[:,i],qvec[i,:]))



        if i == 0:

            if (1):

                R_ = R.cpu().numpy().astype(np.float32)
                T_ = T.cpu().numpy().astype(np.float32)
                text_file = open('./' + outdir + '/epoch_' + epoch_no + '/' + filename_to_save[0][:-4] + '_cam.txt', "w")
                text_file.write('extrinsic\n')
                text_file.write(
                    str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0,i]) + '\n')
                text_file.write(
                    str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1,i]) + '\n')
                text_file.write(
                    str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2,i]) + '\n')
                text_file.write('0' + ' ' + '0' + ' ' + '0' + ' ' + '1' + '\n\n')
                text_file.write('intrinsic\n')
                text_file.write('361.54125' + ' ' + '0' + ' ' + '82.900625\n')
                text_file.write('0' + ' ' + '360.3975' + ' ' + '66.383875\n')
                text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
                text_file.write('425.0' + ' ' + '2.5')
                text_file.close()
            proj_mats += [torch.inverse(proj_mat)]

        else:
            proj_mats += [proj_mat]

    proj_mats = torch.stack(proj_mats)

    proj_mats = proj_mats.unsqueeze(0).cuda() # to make the dimensions like ([1, 3, 4, 4]) from ([3, 4, 4])

    return proj_mats



def read_intrinsics_file(filename, interval_scale):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # intrinsics: line [1-3), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[1:4]), dtype=np.float32, sep=' ')
    #print('lines', lines)
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[5].split()[0])
    depth_interval = float(lines[5].split()[1]) * interval_scale
    #print('intrinsics = {}, depth_min = {}, depth_interval = {}'.format(intrinsics, depth_min, depth_interval))
    return intrinsics, depth_min, depth_interval


def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()


def center_crop(img, dim):
	#Returns center cropped image
	#Args:Image Scaling
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

def inverse_project(intrinsics, qvec, T, depths, kps_src):

    #print("batch size: {}".format(batch_size))
    #print("qvec: {}, T = {}".format(qvec, T))

    #T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image
    #T = T[..., None] # --> transposing a 1D array


    R = torch.zeros([3,3])
    R[0,0] = 1 - 2 * qvec[2]*qvec[2] - 2 * qvec[3]*qvec[3]
    R[0,1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]
    R[0,2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]

    R[1,0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]
    R[1,1] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[3]*qvec[3]
    R[1,2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]

    R[2,0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]
    R[2,1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]
    R[2,2] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[2]*qvec[2]

    proj_mat = np.eye(4)

    proj_mat[:3, :3] = R.numpy()
    T = T.numpy()
    proj_mat[0, 3] = T[0]
    proj_mat[1, 3] = T[1]
    proj_mat[2, 3] = T[2]
    #proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]
    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

    #print("pos_out={} \n, ori_out={} \n".format(T[:,i],qvec[i,:]))
    #print("proj_mat={} ".format(proj_mat))
    depths = depths.cpu().numpy()
    kps_src = kps_src.astype(np.int32)
    #depths = cv2.resize(depths, (256, 256))
    #depths = center_crop(depths, (224, 224))
    #A1 = depths.astype(np.float32)
    #write_pfm('depth_init.pfm',A1)
    #del A1
    #print("Depth shape={} ".format(depths.shape))
    #print("Depth type={} ".format(depths.dtype))
    #depths_ = cv2.normalize(depths, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #cv2.imwrite('depth.png', depths_)
    #print("kps_src={}, kps_src_Type={}".format(kps_src, kps_src.dtype))
    #print('kps_src[j] = {}'.format(kps_src[0]))
    #print('depths[kps_src[j] = {}'.format(depths[kps_src[0,1],kps_src[0,0]]))

    pt2_2D = []

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
        #if(depths[kps_src[j,1],kps_src[j,0]] > 0):
            #print('Point1 2D={}'.format([kps_src[j,1],kps_src[j,0]]))
            #print('Point2 2D={}'.format([pt2[0]/pt2[2], pt2[1]/pt2[2]]))
        pt2_2D.append([pt2[0]/pt2[2], pt2[1]/pt2[2]])

    return pt2_2D


def inverse_project(intrinsics, qvec, T, depths, kps_src):

    #print("batch size: {}".format(batch_size))
    #print("qvec: {}, T = {}".format(qvec, T))

    #T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image
    #T = T[..., None] # --> transposing a 1D array


    R = torch.zeros([3,3])
    R[0,0] = 1 - 2 * qvec[2]*qvec[2] - 2 * qvec[3]*qvec[3]
    R[0,1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]
    R[0,2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]

    R[1,0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]
    R[1,1] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[3]*qvec[3]
    R[1,2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]

    R[2,0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]
    R[2,1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]
    R[2,2] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[2]*qvec[2]

    proj_mat = np.eye(4)

    proj_mat[:3, :3] = R.numpy()
    T = T.numpy()
    proj_mat[0, 3] = T[0]
    proj_mat[1, 3] = T[1]
    proj_mat[2, 3] = T[2]
    #proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]
    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

    #print("pos_out={} \n, ori_out={} \n".format(T[:,i],qvec[i,:]))
    #print("proj_mat={} ".format(proj_mat))
    depths = depths.cpu().numpy()
    kps_src = kps_src.astype(np.int32)
    #depths = cv2.resize(depths, (256, 256))
    #depths = center_crop(depths, (224, 224))
    #A1 = depths.astype(np.float32)
    #write_pfm('depth_init.pfm',A1)
    #del A1
    #print("Depth shape={} ".format(depths.shape))
    #print("Depth type={} ".format(depths.dtype))
    #depths_ = cv2.normalize(depths, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #cv2.imwrite('depth.png', depths_)
    #print("kps_src={}, kps_src_Type={}".format(kps_src, kps_src.dtype))
    #print('kps_src[j] = {}'.format(kps_src[0]))
    #print('depths[kps_src[j] = {}'.format(depths[kps_src[0,1],kps_src[0,0]]))

    pt2_2D = []

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
        #if(depths[kps_src[j,1],kps_src[j,0]] > 0):
            #print('Point1 2D={}'.format([kps_src[j,1],kps_src[j,0]]))
            #print('Point2 2D={}'.format([pt2[0]/pt2[2], pt2[1]/pt2[2]]))
        pt2_2D.append([pt2[0]/pt2[2], pt2[1]/pt2[2]])

    return pt2_2D

def inverse_project3(intrinsics, qvec, T, depths, kps_src):

    R = torch.zeros([3,3])
    R[0,0] = 1 - 2 * qvec[2]*qvec[2] - 2 * qvec[3]*qvec[3]
    R[0,1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]
    R[0,2] = 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]

    R[1,0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]
    R[1,1] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[3]*qvec[3]
    R[1,2] = 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]

    R[2,0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]
    R[2,1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]
    R[2,2] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[2]*qvec[2]

    proj_mat = np.eye(4)

    proj_mat[:3, :3] = R.numpy()
    T = T.numpy()
    proj_mat[0, 3] = T[0]
    proj_mat[1, 3] = T[1]
    proj_mat[2, 3] = T[2]
    proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]

    depths = depths.cpu().numpy()
    kps_src = kps_src.astype(np.int32)
    
    pt2_2D = np.zeros((len(kps_src), 2))

    for j in range(len(kps_src)):
        arr = np.zeros((3,1))
        arr[0] = depths[kps_src[j,1],kps_src[j,0]] * kps_src[j, 0]
        arr[1] = depths[kps_src[j,1],kps_src[j,0]] * kps_src[j, 1]
        arr[2] = depths[kps_src[j,1],kps_src[j,0]]
        
        pt1_3D = (np.linalg.inv(intrinsics) @ arr)
       
        arr2 = np.ones((4, 1))
        arr2[:3] = pt1_3D
       
        pt2 = proj_mat[:3, :4] @ arr2
       
        pt2_2D[j, 0] = pt2[0]/pt2[2]
        pt2_2D[j, 1] = pt2[1] / pt2[2]

    return pt2_2D

# 2d-2d 투영
# projection matrix는 https://ichi.pro/ko/homography-matrix-chujeong-90350826572588 을따름
def inverse_project4(intrinsics, qvec, T, kps_src):
    
    R = torch.zeros([3,3])
    R[0,0] = 1 - 2 * qvec[2]*qvec[2] - 2 * qvec[3]*qvec[3]
    R[0,1] = 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3]
    R[0,2] = T[0]

    R[1,0] = 2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3]
    R[1,1] = 1 - 2 * qvec[1]*qvec[1] - 2 * qvec[3]*qvec[3]
    R[1,2] = T[1]

    R[2,0] = 2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2]
    R[2,1] = 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1]
    R[2,2] = T[2]

    proj_mat = np.eye(3)

    proj_mat[:3, :3] = R.numpy()
    T = T.numpy()

    proj_mat[:3, :3] = intrinsics @ proj_mat[:3, :3]

    kps_src = kps_src.astype(np.int32)
    
    pt2_2D = np.zeros((len(kps_src), 2))

    for j in range(len(kps_src)):
        arr = np.ones((3,1))
        arr[0] = kps_src[j, 0]
        arr[1] = kps_src[j, 1]
        pt2_2D[j][0] = (proj_mat @ arr)[0]
        pt2_2D[j][1] = (proj_mat @ arr)[1]

    return pt2_2D




# for printing cam files
def RT_to_ProjectionMartices_visheatmap(intrinsics, qvec, T):

    T = torch.transpose(T, 0, 1) # --> each column of the T represent translation of that image


    R = torch.zeros([3,3])
    R[0,0] = 1 - 2 * qvec[0,2]*qvec[0,2] - 2 * qvec[0,3]*qvec[0,3]
    R[0,1] = 2 * qvec[0,1] * qvec[0,2] - 2 * qvec[0,0] * qvec[0,3]
    R[0,2] = 2 * qvec[0,3] * qvec[0,1] + 2 * qvec[0,0] * qvec[0,2]

    R[1,0] = 2 * qvec[0,1] * qvec[0,2] + 2 * qvec[0,0] * qvec[0,3]
    R[1,1] = 1 - 2 * qvec[0,1]*qvec[0,1] - 2 * qvec[0,3]*qvec[0,3]
    R[1,2] = 2 * qvec[0,2] * qvec[0,3] - 2 * qvec[0,0] * qvec[0,1]

    R[2,0] = 2 * qvec[0,3] * qvec[0,1] - 2 * qvec[0,0] * qvec[0,2]
    R[2,1] = 2 * qvec[0,2] * qvec[0,3] + 2 * qvec[0,0] * qvec[0,1]
    R[2,2] = 1 - 2 * qvec[0,1]*qvec[0,1] - 2 * qvec[0,2]*qvec[0,2]

    proj_mat = torch.eye(4)

    proj_mat[:3, :3] = R
    proj_mat[:3, 3] = T[:,0]
    proj_mat[:3,:4] = (torch.from_numpy(intrinsics))@proj_mat[:3,:4]

        #print("pos_out={} \n, ori_out={} \n".format(T[:,i],qvec[i,:]))


    return proj_mat


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

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)



def homo_warping(src_fea, src_proj, ref_proj):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    src_fea = np.array(src_fea)
    print(src_fea)
    channels = src_fea.shape[2]
    height, width = src_fea.shape[0], src_fea.shape[1]
    new_feat = np.zeros(src_fea.shape)
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        for h in range(height):
            for w in range(width):
                new_point = rot @ np.array([w,h,1])
                new_point = new_point.int()
                for c in range(channels):

                    value = src_fea[h][w][c]
                    new_x = new_point[0][0] * -1
                    new_y = new_point[0][1] * -1
            
                    if new_x > 639 or new_x < 0 or new_y > 511 or new_y <0:
                        continue
                    new_feat[new_y, new_x, c] = value
                    
                    
    return new_feat
