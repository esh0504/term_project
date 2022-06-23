# run the inference of RTNet for 3D reconstruction purpose
# Relative RT experiments with Siamese Network
# AFter inference, the ntework will generate the _cam.txt files that can be renamed and used in the reconstruction experiments

#Inference using RTNet network
CUDA_VISIBLE_DEVICES=0 python RTNet_Zahid_v8_2.py --root_dir=dummy_Data/ --n_views=8 --n_depths=192 --interval_scale=1.06 --batch_size=1 --start_epoch=23 --num_epochs=1 --exp_name=rel_RT55 --train_from_scratch=False --testMVSNet=True --optimizer='adam' --dropout_rate=0.3 --bayesian=True --warmup_epochs=1 --lr=0.001 --sum_mode=True --usePercenLoss=False --modelRT_load_path=ckpts/Zahid_RT/rel_RT55/ckpts_23.model --modelMVSNet_load_path=ckpts/_ckpt_epoch_5.ckpt

#Note that the --n_views=9 will work if you use the pair_zahid2.txt (update it in the dataset/dtu_both_relative8.py --> line --> pair_file = "Cameras/pair_zahid2.txt"), because in pair_zahid2.txt, every line contains 8 pairs, and the pair_zahid2.txt covers all the 49 views
#Also note that the depth inference from MVSNet is currently disabled because I was experimenting with the ground truth depths. But the depth inference from MVSNEt can simply be enabled by uncommenting the corresponding lines in def test_MVSNet function.
