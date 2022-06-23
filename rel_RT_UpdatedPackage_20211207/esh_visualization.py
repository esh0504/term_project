# Combined imports
import os
from opt import get_opts

args = get_opts()

test_data = args.test_data
epoch = args.epoch
test_scan_num = args.test_scan_num

test_data_save_path = test_data + '/epoch_' + epoch + '/testset/' 
test_data_path = test_data_save_path + '/scan' + test_scan_num + '_train_rect_001_0_r5000_'


# if not os.path.exists(test_data_save_path + 'cams'):
#         os.makedirs(test_data_save_path + 'cams')
        

# if not os.path.exists(test_data_save_path + 'depths'):
#         os.makedirs(test_data_save_path + 'depths')
        
if not os.path.exists(test_data_save_path + 'rectified'):
        os.makedirs(test_data_save_path + 'rectified')
        
# os.system('cp ' + test_data_path + '* ' + test_data_save_path + 'cams')
# os.system('cp ' + test_data_save_path + '*.pfm ' + test_data_save_path + 'depths') 
os.system('cp ' + test_data_save_path + '*.png ' + test_data_save_path + 'rectified') 

# file_names = os.listdir(test_data_save_path + 'cams')
# print(len(file_names))
# for file_name in file_names:
#     src = os.path.join(test_data_save_path + 'cams/', file_name)
#     dst = '0000'+ file_name[-11:]
#     dst = os.path.join(test_data_save_path + 'cams/', dst)
#     os.rename(src, dst)
    
# file_names = os.listdir(test_data_save_path + 'depths')
# print(len(file_names))
# for file_name in file_names:
#     src = os.path.join(test_data_save_path + 'depths/', file_name)
#     dst = '0000'+ file_name[18:21] + file_name[29:]
#     dst = os.path.join(test_data_save_path + 'depths/', dst)
#     os.rename(src, dst)
    
file_names = os.listdir(test_data_save_path + 'rectified')
print(len(file_names))
for file_name in file_names:
    src = os.path.join(test_data_save_path + 'rectified/', file_name)
    dst = '0000'+ file_name[-15:-12]+'.png'
    dst = os.path.join(test_data_save_path + 'rectified/', dst)
    os.rename(src, dst)

