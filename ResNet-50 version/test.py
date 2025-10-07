import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from data import test_dataset, eval_Dataset
from tqdm import tqdm
# from model.DPT import DPTSegmentationModel
from config import param as option
from model.get_model import get_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.autograd import Variable



def eval_mae(loader, cuda=True):
    avg_mae, img_num, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in loader:
            if cuda:
                pred, gt = trans(pred).cuda(), trans(gt).cuda()
            else:
                pred, gt = trans(pred), trans(gt)
            mae = torch.abs(pred - gt).mean()
            if mae == mae: # for Nan
                avg_mae += mae
                img_num += 1.0
        avg_mae /= img_num
    return avg_mae


# Begin the testing process
_, swin_model = get_model(option)
swin_model.load_state_dict(torch.load(option['ckpt_save_path']+'/50_semi_model.pth'))
# generator.cuda()
swin_model.eval()

test_datasets, pre_root = option['datasets'], option['eval_save_path']

time_list, mae_list = [], []
test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]

save_path_base = pre_root + test_epoch_num + '_ebm/'

# save_path_base = pre_root
# Begin to inference and save masks
print('========== Begin to inference and save masks ==========')
def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 12, 12]).to(device)
index = 0


for dataset in test_datasets:
    
    save_path_mean_swin_body = './results_mean_swin_body/' + dataset + '/'
    if not os.path.exists(save_path_mean_swin_body):
        os.makedirs(save_path_mean_swin_body)
    
    save_path_mean_swin_contour = './results_mean_swin_contour/' + dataset + '/'
    if not os.path.exists(save_path_mean_swin_contour):
        os.makedirs(save_path_mean_swin_contour)
    
    save_path_mean_swin = './results_mean_swin/' + dataset + '/'
    if not os.path.exists(save_path_mean_swin):
        os.makedirs(save_path_mean_swin)
    
    
    image_root = ''
    # depth_root = ''
    # if option['task'] == 'SOD':
    image_root = option['test_dataset_root'] + dataset + '/'
    # elif option['task'] == 'RGBD-SOD':
    #     image_root = option['test_dataset_root'] + dataset + '/RGB/'
    #     depth_root = option['test_dataset_root'] + dataset + '/depth/'

    test_loader = test_dataset(image_root, option['testsize'])
    #for iter in range(9):
    for i in tqdm(range(test_loader.size), desc=dataset):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        swin_mean_body = 0
        swin_mean_contour = 0
        swin_mean_pred = 0
        
    
        torch.cuda.synchronize()
        start = time.time()

        for iter in range(option['iter_num']):        
            pred_swin_body, pred_swin_contour, pred_swin_pred  = swin_model.forward(image, tag="test")  # Inference and get the last one of the output list

            swin_mean_body = swin_mean_body + torch.sigmoid(pred_swin_body)
            swin_mean_contour = swin_mean_contour + torch.sigmoid(pred_swin_contour)
            swin_mean_pred = swin_mean_pred + torch.sigmoid(pred_swin_pred)
           
        swin_mean_body = swin_mean_body / option['iter_num']
        swin_mean_contour = swin_mean_contour / option['iter_num']
        swin_mean_prediction = swin_mean_pred / option['iter_num']
        
        res_body = F.upsample(swin_mean_body, size=[WW, HH], mode='bilinear', align_corners=False)
        res_body = res_body.data.cpu().numpy().squeeze()
        res_body = 255 * (res_body - res_body.min()) / (res_body.max() - res_body.min() + 1e-8)
        cv2.imwrite(save_path_mean_swin_body + name, res_body)
        
        res_contour = F.upsample(swin_mean_contour, size=[WW, HH], mode='bilinear', align_corners=False)
        res_contour = res_contour.data.cpu().numpy().squeeze()
        res_contour = 255 * (res_contour - res_contour.min()) / (res_contour.max() - res_contour.min() + 1e-8)
        cv2.imwrite(save_path_mean_swin_contour + name, res_contour)
      
        res = F.upsample(swin_mean_prediction, size=[WW, HH], mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path_mean_swin + name, res)
      
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - start)
    print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))

# # Begin to evaluate the saved masks
# print('========== Begin to evaluate the saved masks ==========')
# for dataset in tqdm(test_datasets):
#     if option['task'] == 'COD':
#         gt_root = option['test_dataset_root'] + dataset + '/GT'
#     else:
#         gt_root = option['test_dataset_root'] + '/GT/' + dataset + '/'
#
#     loader = eval_Dataset(os.path.join(save_path_base, dataset), gt_root)
#     mae = eval_mae(loader=loader, cuda=True)
#     mae_list.append(mae.item())
#
# print('--------------- Results ---------------')
# results = np.array(mae_list)
# results = np.reshape(results, [1, len(results)])
# mae_table = pd.DataFrame(data=results, columns=test_datasets)
# with open(save_path_base+'results.csv', 'w') as f:
#     mae_table.to_csv(f, index=False, float_format="%.4f")
# print(mae_table.to_string(index=False))
# print('--------------- Results ---------------')
