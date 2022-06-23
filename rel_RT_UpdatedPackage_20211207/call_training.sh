#run the training using the shell file
# Relative RT experiments with Siamese Network

#Fine-tune the RTNet network
# CUDA_VISIBLE_DEVICES=0 python RTNet_esh_v8_2_with_SURF.py --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=18 --num_epochs=100 --exp_name=rel_RT1 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --modelRT_load_path=./ckpts/esh_RT_SURF/rel_RT1/ckpts_17.model
# CUDA_VISIBLE_DEVICES=1 python RTNet_esh_v8_2_with_SIFT.py --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=92 --num_epochs=100 --exp_name=rel_RT1 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --modelRT_load_path=./ckpts/esh_RT_SIFT/rel_RT1/ckpts_91.model
CUDA_VISIBLE_DEVICES=0 python RTNet_Zahid_v8_2.py --base_model=resnext50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=4 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --modelRT_load_path=./ckpts/esh_RT/rel_RT55_resnext50/ckpts_3.model
CUDA_VISIBLE_DEVICES=1 python RTNet_Zahid_v8_2_esh.py --base_model=resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False 
CUDA_VISIBLE_DEVICES=2 python RTNet_Zahid_v8_2.py --base_model=wide_resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False 
CUDA_VISIBLE_DEVICES=0 python RTNet_Zahid_v8_2_esh.py --base_model=wide_resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False 
CUDA_VISIBLE_DEVICES=1 python RTNet_Zahid_v8_2.py --base_model=resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=100 --num_epochs=200 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=True --modelRT_load_path=/data_hdd/SeungHo/model_ckpt/RTNet/rel_RT55_resnet50_feat/ckpts_100.model
CUDA_VISIBLE_DEVICES=2 python RTNet_Zahid_v8_2_esh.py --base_model=resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False 
CUDA_VISIBLE_DEVICES=1 python RTNet_Zahid_v8_2_esh.py --base_model=wide_resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False 
CUDA_VISIBLE_DEVICES=0 python RTNet_Zahid_v8_2_esh.py --base_model=resnext50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=2 --start_epoch=1 --num_epochs=15 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --loss_method=diff_square

#Train from scratch the RTNet
#CUDA_VISIBLE_DEVICES=0 python RTNet_Zahid_v8_2.py --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=8 --start_epoch=1 --num_epochs=100 --exp_name=rel_RT55 --train_from_scratch=True --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False

#Please use pair_zahid4.txt at the training time to use all the combinations of overlapping pairs in DTUdataset (update it in the dataset/dtu_both_relative8.py --> line --> pair_file = "Cameras/pair_zahid4.txt"), because in pair_zahid4.txt, every line contains 7 pairs, and the pair_zahid4.txt covers all the 49 views


CUDA_VISIBLE_DEVICES=1 python RTNet_Zahid_v8_2.py --base_model=resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=1 --num_epochs=100 --exp_name=rel_RT55_L2 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False
# perc_loss 
CUDA_VISIBLE_DEVICES=1 python RTNet_Zahid_v8_2.py --base_model=resnet50 --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=100 --num_epochs=200 --exp_name=rel_RT55 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=True --modelRT_load_path=/data_hdd/SeungHo/model_ckpt/RTNet/rel_RT55_resnet50_feat/ckpts_100.model