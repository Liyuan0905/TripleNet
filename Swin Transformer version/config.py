import os
import time
import argparse

parser = argparse.ArgumentParser(description='Decide Which Task to Training')
parser.add_argument('--task', type=str, default='SOD', choices=['SOD','RGBD-SOD'])
parser.add_argument('--model', type=str, default='swin',choices=['swin'])
# parser.add_argument('--training_path', type=str, default='D:/Code/Data/data_sod/RGB_Dataset/train/DUTS')
parser.add_argument('--training_path', type=str, default='/raid/datasets/data_sod/RGB_Dataset/train/train/DUTS/')
# parser.add_argument('--training_path', type=str, default='/media/data1/data_sod/RGB_Dataset/train/DUTS/')
# parser.add_argument('--training_path', type=str, default='/home/cly/data_sod/RGB_Dataset/train/DUTS/')

parser.add_argument('--log_info', type=str, default='REMOVE')
parser.add_argument('--ckpt', type=str, default='SOD')    
args = parser.parse_args()

# Configs
param = {}
param['task'] = args.task

# Training Config
param['partial_ratio'] = 1/32 
# param['masked_ratio'] = 0.1
param['spv_epoch'] = 30          # max epoch
param['epoch'] = 50          # max epoch
param['seed'] = 1234          # random seeds
param['batch_size'] = 8    # batch size
param['spv_save_epoch'] = 30       # snap shoots
param['save_epoch'] = 50       # snap shoots
param['lr'] = 6e-5          # learning rate
param['lr_dis'] = 1e-5          # learning rate
param['trainsize'] = 384      # training image size
param['patchsize'] = 16      # masked patch size
param['decay_rate'] = 0.5
param['spv_decay_epoch'] = 20
param['decay_epoch'] = 30
param['beta'] = [0.5, 0.999]  # Adam related parameters
param['size_rates'] = [1]     # Multi-scale  [0.75, 1, 1.25]/[1]
param['use_pretrain'] = True
param['attention_decoder'] = True
## ABP related
param['latent_dim'] = 32
param['langevin_step_num_gen'] = 5
param['sigma_gen'] = 0.3
param['langevin_s'] = 0.1
## EBM related
param['ebm_out_dim'] = 1
param['ebm_middle_dim'] = 60
param['e_init_sig'] = 1.0
param['e_l_steps'] = 5
# param['e_l_steps'] = 80
param['e_l_step_size'] = 0.4
param['e_prior_sig'] = 1.0
param['g_l_steps'] = 5
# param['g_l_steps'] = 40
param['g_llhd_sigma'] = 0.3
param['g_l_step_size'] = 0.1
param['e_energy_form'] = 'identity'

# Backbone Config
param['model_name'] = args.model   # [VGG, ResNet, DPT]
param['backbone_name'] = "vitb_rn50_384"   # vitl16_384

param['iter_num'] = 1

# Dataset Config
if param['task'] == 'SOD':
    param['image_root'] = args.training_path + '/img/'
    param['gt_root'] = args.training_path + '/gt/'
    
    param['psd_root'] = './res_pred/'
    param['edge_root'] = args.training_path + '/edge/'
    param['psd_contour_root'] = './res_contour/'
    param['psd_body_root'] = './res_body/'
    
    # param['test_dataset_root'] = '/media/circle/SeagateExp/data_sod/RGB_Dataset/test/img/'
    # param['test_dataset_root'] = '/media/data1/data_sod/RGB_Dataset/test/img/'
    param['test_dataset_root'] = '/raid/datasets/data_sod/RGB_Dataset/test/test/img/'
    # param['test_dataset_root'] = '/home/cly/data_sod/RGB_Dataset/test/img/'
    # param['test_dataset_root'] = './RGB_Dataset/test/img/'
elif param['task'] == 'RGBD-SOD':
    param['image_root'] = './RGBD_Dataset/train/old_train/RGB/'
    param['gt_root'] = './RGBD_Dataset/train/old_train/GT/'
    param['depth_root'] = './RGBD_Dataset/train/old_train/depth/'
    param['test_dataset_root'] = './RGBD_Dataset/test/'


# Experiment Dir Config
log_info = args.model + '_' + args.log_info    #
param['training_info'] = param['task'] + '_' + param['backbone_name'] + '_' + str(param['lr']) + '_' + log_info
param['spv_log_save_path'] = 'experiments/{}/spv/'.format(param['training_info'])   #
param['semi_log_save_path'] = 'experiments/{}/semi/'.format(param['training_info'])   #

param['spv_log_path'] = 'experiments/{}'.format(param['training_info'])   #
param['semi_log_path'] = 'experiments/{}'.format(param['training_info'])   #

param['ckpt_save_path'] = param['spv_log_path'] + '/models/'              #
param['ckpt_save_path'] = param['semi_log_path'] + '/models/'              #

print('[INFO] Experiments saved in: ', param['training_info'])


# Test Config
param['testsize'] = 384
param['checkpoint'] = args.ckpt
param['eval_save_path'] = param['spv_log_path'] + '/save_images/'         #
param['eval_save_path'] = param['semi_log_path'] + '/save_images/'         #

if param['task'] == 'SOD':
    param['datasets'] = ['DUTS', 'ECSSD', 'DUT', 'HKU-IS', 'PASCAL', 'SOD']
elif param['task'] == 'RGBD-SOD':
    param['datasets'] = ['DES', 'LFSD', 'NJU2K', 'NLPR','SIP','STERE']

