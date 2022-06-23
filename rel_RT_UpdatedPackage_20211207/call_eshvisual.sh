#run the training using the shell file
# Relative RT experiments with Siamese Network

#Fine-tune the RTNet network
# CUDA_VISIBLE_DEVICES=0 python RTNet_esh_v8_2_with_SURF.py --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=18 --num_epochs=100 --exp_name=rel_RT1 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --modelRT_load_path=./ckpts/esh_RT_SURF/rel_RT1/ckpts_17.model
# CUDA_VISIBLE_DEVICES=1 python RTNet_esh_v8_2_with_SIFT.py --root_dir=../datasets/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=4 --start_epoch=92 --num_epochs=100 --exp_name=rel_RT1 --train_from_scratch=False --trainRT_net=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --sum_mode=True --usePercenLoss=False --modelRT_load_path=./ckpts/esh_RT_SIFT/rel_RT1/ckpts_91.model
python visHeatMap.py --base_model=resnext50 --start_epoch=10 --exp_name=rel_RT55