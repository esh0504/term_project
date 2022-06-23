import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/gpuadmin/zahid_data/MVSNet_pl/MVSNet_pl/datasets/train_data/dtu/',
                        help='root directory of dtu dataset')
    parser.add_argument('--n_views', type=int, default=3,
                        help='number of views (including ref) to be used in training')
    parser.add_argument('--n_depths', type=int, default=256,
                        help='number of depths of cost volume')
    parser.add_argument('--interval_scale', type=float, default=0.8,
                        help='depth interval scale between each depth step (2.5mm)')
    
    parser.add_argument('--loss_type', type=str, default='sl1',
                        choices=['sl1'],
                        help='loss to use')
    
    parser.add_argument('--base_model', type=str, default='resnet50',
                        help='base model for siamese Net')
    
    parser.add_argument('--loss_method', type=str, default='diff',
                        help='loss_method for RT')
    
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=6,
                        help='number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='starting number of the epoch')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default='ckpts/casmvsnet.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--ckpt_path_RT', type=str, default='./Umar/Network_90.pth',
                        help='pretrained checkpoint path to load the RT Network')
                        
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
                        
    ###########################
    #### params for Umar RT ####
    parser.add_argument('--sum_mode', type=bool, default=False,
                        help='To be explained by Umar')
                        
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='To be explained by Umar')
                        
    parser.add_argument('--bayesian', type=bool, default=False,
                        help='To be explained by Umar')

    parser.add_argument('--img_size', type=int, default=256,
                        help='image size to train the RT network')

    parser.add_argument('--img_crop', type=int, default=224,
                        help='image crop to train the RT network')
                        
    parser.add_argument('--trainRT_net', type=bool, default=False,
                        help='if want to Train RT net')

    parser.add_argument('--testRT_net', type=bool, default=False,
                        help='if want to Test RT net')

    parser.add_argument('--modelRT_load_path', type=str, default="./Umar/Network_90.pth",
                        help='if want to Train RT net')

    parser.add_argument('--train_from_scratch', type=bool, default=False,
                        help='if want to Train network from scratches')

    parser.add_argument('--modelRT_save_path', type=str, default="",
                        help='default patth to save RT net checkpoints')

    parser.add_argument('--modelRT_Reg_load_path', type=str, default="",
                        help='default patth to save RT net checkpoints')
                        
    parser.add_argument('--trainMVSNet', type=bool, default=False,
                        help='if want to Train MVSNet net')

    parser.add_argument('--freezeMVSNet', type=bool, default=False,
                        help='if want to Freeze MVSNet net (only use as feature extractor)')
                        
    parser.add_argument('--testMVSNet', type=bool, default=False,
                        help='if want to Test MVSNet net with pre-trained RT')
                        
    parser.add_argument('--modelMVSNet_save_path', type=str, default="./MVSNet_models",
                        help='default patth to save MVSNet checkpoints')

    parser.add_argument('--modelMVSNet_load_path', type=str, default="./MVSNet_models",
                        help='path to load the MVSNet checkpoints')

    parser.add_argument('--intrinsic_file_path', type=str, default="./intrinsics.txt",
                        help='path to intrinsics file')
                        
    parser.add_argument('--trainBoth', type=bool, default=False,
                        help='if want to Train Both RT net and MVSNet net')

    parser.add_argument('--usePercenLoss', type=bool, default=False,
                        help='if want to Train/fintune RT net using percentage loss')
    
    parser.add_argument('--loss_feat', type=bool, default=False,
                        help='want + zahid\'s loss_feat')
    
    
    ###########################
    
    # esh_visualization
    
    parser.add_argument('--test_data', type=str, default=False,
                        help='if want to Train/fintune RT net using percentage loss')
    parser.add_argument('--epoch', type=str, default=False,
                        help='if want to Train/fintune RT net using percentage loss')
    parser.add_argument('--test_scan_num', type=str, default=False,
                        help='if want to Train/fintune RT net using percentage loss')
    parser.add_argument('--visual_epoch', type=int, default=3,
                        help='if want to Train/fintune RT net using percentage loss')
    
    parser.add_argument('--MVSTest', type=int, default=0,
                        help='if want to Train/fintune RT net using percentage loss')
    return parser.parse_args()
