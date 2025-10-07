import cv2
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
swin_model, _ = get_model(option)
swin_model.load_state_dict(torch.load(option['ckpt_save_path']+'/30_spv_model.pth'))
# generator.cuda()
swin_model.eval()

test_datasets = option["image_root"]

time_list, mae_list = [], []
test_epoch_num = option['checkpoint'].split('/')[-1].split('_')[0]


save_path_mean_swin_body = './res_body/' 
if not os.path.exists(save_path_mean_swin_body):
    os.makedirs(save_path_mean_swin_body)

save_path_mean_swin_contour = './res_contour/'
if not os.path.exists(save_path_mean_swin_contour):
    os.makedirs(save_path_mean_swin_contour)

save_path_mean_swin = './res_pred/'
if not os.path.exists(save_path_mean_swin):
    os.makedirs(save_path_mean_swin)


image_root = option['image_root']  

test_loader = test_dataset(image_root, option['testsize'])
for i in tqdm(range(test_loader.size)):
    image, HH, WW, name = test_loader.load_data()
    image = image.cuda()    
    torch.cuda.synchronize()
    start = time.time()

    pred_swin_body, pred_swin_contour, pred_swin_pred  = swin_model.forward(image, tag = "test")  # Inference and get the last one of the output list
 
    swin_mean_body = torch.sigmoid(pred_swin_body)
    swin_mean_contour = torch.sigmoid(pred_swin_contour)
    swin_mean_pred = torch.sigmoid(pred_swin_pred)
    

    res_body = F.upsample(swin_mean_body, size=[WW, HH], mode='bilinear', align_corners=False)
    res_body = res_body.data.cpu().numpy().squeeze()
    res_body = 255 * (res_body - res_body.min()) / (res_body.max() - res_body.min() + 1e-8)
    cv2.imwrite(save_path_mean_swin_body + name, res_body)
    
    res_contour = F.upsample(swin_mean_contour, size=[WW, HH], mode='bilinear', align_corners=False)
    res_contour = res_contour.data.cpu().numpy().squeeze()
    res_contour = 255 * (res_contour - res_contour.min()) / (res_contour.max() - res_contour.min() + 1e-8)
    cv2.imwrite(save_path_mean_swin_contour + name, res_contour)
    
    res = F.upsample(swin_mean_pred, size=[WW, HH], mode='bilinear', align_corners=False)
    res = res.data.cpu().numpy().squeeze()
    res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(save_path_mean_swin + name, res)
    


    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - start)
print('[INFO] Avg. Time used in this sequence: {:.4f}s'.format(np.mean(time_list)))

