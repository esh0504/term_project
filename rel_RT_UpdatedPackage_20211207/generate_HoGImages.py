import os
import numpy as np
import cv2 as cv
from skimage.feature import hog
from PIL import Image, ImageFont, ImageDraw

boundary_away = 20

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


def Get_HOG_Descriptor(image):
    # image = np.array(image) #convert PIL image to numpy array (cv2 Mat)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (1200 // 4, 1600 // 4))
    # resize_img = image.resize((224, 224))

    # (H, hogImage) = hog(resize_img, orientations=9, pixels_per_cell=(16, 16),
    #                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    (H, hogImage) = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True,
                        block_norm="L1", visualize=True)
    # hogImage = hogImage.resize((640, 512))
    # print ("size of H : {}, size of hogImage : {}".format(H.shape, hogImage.shape))
    # return torch.FloatTensor(H)
    return hogImage


dtu_dataset_folder = '../../../../home/gpuadmin/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/Rectified/'
HoGImgs_folder = './HoGImgs_folder/'

if not os.path.exists(HoGImgs_folder):
    os.mkdir(HoGImgs_folder)

folder_names = os.listdir(dtu_dataset_folder) #get the foldernames in dtu dataset

size_of_dir = len(folder_names)
file_count = 1
for folder_name in folder_names: # get the filenames in each folder of dtu dataset
    if not os.path.exists(HoGImgs_folder + folder_name):
        os.mkdir(HoGImgs_folder + folder_name)

    filenames = os.listdir(dtu_dataset_folder + folder_name + '/')
    file_count += 1

    for filename in filenames: # for each file in dataset, find SIFT features and save in a txt file
        if (filename.endswith('.png')):
            img_filename = os.path.join(dtu_dataset_folder,folder_name,filename)
            img = Image.open(img_filename)
            img_cpy = img.copy()

            HoG_img = Get_HOG_Descriptor(img_cpy)  # compute the HoG features before transforming the image

            # print("Hog image type : {}".format(HoG_img.dtype))

            # (Image.fromarray(np.uint8(HoG_img * 255))).save("HogImage.png")

            red, green, blue = img_cpy.split()

            # red.save("Red.png")

            # print('red type : {}'.format(red.dtype))
            # img_cpy = (Image.merge("RGB", [red, green, HoG_img])).save("merged_img.png")
            # (Image.merge("RGB", [red, green, Image.fromarray(np.uint8(HoG_img * 255))])).save("merged_img.png")
            img_cpy = Image.merge("RGB", [red, green, Image.fromarray(np.uint8(HoG_img * 255))])

            img_cpy.save(os.path.join(HoGImgs_folder,folder_name,filename))

            #quit()

    print('Files done : {} of {} \n'.format(file_count, size_of_dir))



