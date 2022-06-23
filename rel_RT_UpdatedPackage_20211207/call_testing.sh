# run the testing of RTNet for evaluation purpose
# Relative RT experiments with Siamese Network

#Test the RTNet network
CUDA_VISIBLE_DEVICES=0,1 python RTNet_esh_v8_2_with_SURF.py --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=1 --start_epoch=75 --num_epochs=1 --exp_name=rel_RT1 --train_from_scratch=False --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=ckpts/esh_RT_SURF/rel_RT1/ckpts_74.model --ckpt_path=ckpts/casmvsnet.ckpt --testMVSNet=True 
CUDA_VISIBLE_DEVICES=0,1 python RTNet_esh_v8_2_with_SIFT.py --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=1 --start_epoch=92 --num_epochs=1 --exp_name=rel_RT1 --train_from_scratch=False --testRT_net=False --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=ckpts/esh_RT_SIFT/rel_RT1/ckpts_91.model --ckpt_path=ckpts/casmvsnet.ckpt --testMVSNet=True 
CUDA_VISIBLE_DEVICES=0,1 python RTNet_Zahid_v8_2.py --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=1 --start_epoch=3 --num_epochs=3 --exp_name=rel_RT55 --train_from_scratch=False --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=ckpts/esh_RT/rel_RT55_resnet50/ckpts_3.model --ckpt_path=ckpts/casmvsnet.ckpt --testMVSNet=True 
CUDA_VISIBLE_DEVICES=2 python RTNet_Zahid_v8_2.py  --base_model=resnet50 --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=1 --num_epochs=1 --exp_name=rel_RT55 --train_from_scratch=False --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=ckpts/esh_RT/rel_RT55_resnet50/ckpts_150.model 

# cam.txt 저장하는 명령, 변경할 값 :(start_epoch, base_model, modelRT_load_path)
CUDA_VISIBLE_DEVICES=2 python RTNet_Zahid_v8_2_esh.py  --base_model=resnet50 --root_dir=/data_hdd/SeungHo/dtu_training/mvs_training/dtu --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=1 --start_epoch=84 --num_epochs=1 --exp_name=rel_RT55 --train_from_scratch=False --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=/data_hdd/SeungHo/model_ckpt/RTNet/rel_RT55_L2_mean_resnet50/ckpts_84.model --testMVSNet=True --MVSTest=0 

#Note that the --n_views=9 will work if you use the pair_zahid2.txt (update it in the dataset/dtu_both_relative8.py --> line --> pair_file = "Cameras/pair_zahid2.txt"), because in pair_zahid2.txt, every line contains 8 pairs, and the pair_zahid2.txt covers all the 49 views