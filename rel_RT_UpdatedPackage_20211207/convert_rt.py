import os
import numpy as np

ref_mat = np.array([[0.9998199, 0.018963303, 0.00079930754, 3.3888612],
[-0.018944856, 0.9996425, -0.018866641, 3.405372],
[-0.0011567955, 0.0188481, 0.99982166, 0.8977877],
[0, 0 ,0, 1]])

indir = '/home/gpuadmin/Seungho/rel_RT_UpdatedPackage_20211207/results_prediction/pr_ep100_rel_RT55.txt'
outdir = './trainRTplusMVSNET/scan10_rel3/cams/'


for filenames in os.listdir(indir):

	with open(indir+filenames) as f:
            lines = [line.rstrip() for line in f.readlines()]

	src_mat = np.zeros((4,4))
	for y in range(4):
		for x in range(4):
			src_mat[y,x] = float(lines[y+1].split(' ')[x])

	R_ = (np.linalg.inv(ref_mat[:3,:3])).dot(src_mat[:3,:3]) # multiply inverse of ref rotation with rotation of src
#	T_ = (np.linalg.inv(ref_mat[:3,:3])).dot(src_mat[:3,3] - ref_mat[:3,3]) # relative translation
	T_ = (src_mat[:3,:3]).dot( - np.transpose(src_mat[:3,:3]).dot(ref_mat[:3,3])) + src_mat[:3,3] # relative translation

#	filename_out = filenames[:-8] + '.txt' # dpeths_mvsnet folder
	filename_out = filenames # cam folder
        
	text_file = open('./' + outdir + filename_out, "w")
	# extrinsics: line [1,5), 4x4 matrix
	text_file.write(lines[0] + '\n')        
	text_file.write(str(R_[0,0]) + ' ' + str(R_[0,1]) + ' ' + str(R_[0,2]) + ' ' + str(T_[0]) + '\n')
	text_file.write(str(R_[1,0]) + ' ' + str(R_[1,1]) + ' ' + str(R_[1,2]) + ' ' + str(T_[1]) + '\n')
	text_file.write(str(R_[2,0]) + ' ' + str(R_[2,1]) + ' ' + str(R_[2,2]) + ' ' + str(T_[2]) + '\n')
	text_file.write(lines[4] + '\n') 

	text_file.write('\n\n')
	text_file.write('intrinsic\n')
#	text_file.write('2892.33' + ' ' + '0' + ' ' + '823.205\n')
#	text_file.write('0' + ' ' + '2883.18' + ' ' + '619.071\n')
	text_file.write('361.54125' + ' ' + '0' + ' ' + '82.900625\n')
	text_file.write('0' + ' ' + '361.54125' + ' ' + '66.383875\n')
	text_file.write('0' + ' ' + '0' ' ' + '1\n\n')
#	text_file.write('425.0' + ' ' + '2.5')
	text_file.write('425.0' + ' ' + '2.5' + ' ' + '192.0' + ' ' + '933.8')
	text_file.close()
	f.close()
