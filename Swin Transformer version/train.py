import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import cv2
import time
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from tqdm import tqdm
from data import get_loader, get_loader_rgbd, get_loader_weak, SalObjDatasetRGB, test_dataset
# from data_semi import SalImgDatasetRGB, SalGTDatasetRGB, SalObjDatasetRGB
from torch.utils.data import random_split
from img_trans import scale_trans
from config import param as option
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import AvgMeter, set_seed, visualize_all, adjust_lr, adjust_uspv_lr
from model.get_model import get_model
from loss.get_loss import get_loss, cal_loss
from img_trans import rot_trans, scale_trans
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mse_loss = torch.nn.MSELoss(reduction='sum')
# mse_loss = torch.nn.MSELoss(reduction='sum')
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
kl_div = torch.nn.KLDivLoss(reduction='none')
bce_loss = torch.nn.CrossEntropyLoss(reduction="none")

mse = torch.nn.MSELoss(None)
cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-4)

import loss.smoothness
smooth_loss = loss.smoothness.smoothness_loss(size_average = True)

from loss.StructureConsistency import SaliencyStructureConsistency
import random

def energy(score):
    if option['e_energy_form'] == 'tanh':
        energy = F.tanh(score.squeeze())
    elif option['e_energy_form'] == 'sigmoid':
        energy = F.sigmoid(score.squeeze())
    elif option['e_energy_form'] == 'softplus':
        energy = F.softplus(score.squeeze())
    else:
        energy = score.squeeze()
    return energy

def get_optim_swin(option, params):
    optimizer = torch.optim.Adam(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['spv_decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler

def get_optim_cnn(option, params):
    optimizer = torch.optim.Adam(params, option['lr'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['spv_decay_epoch'], gamma=option['decay_rate'])
16

def get_optim_dis(option, params):
    optimizer = torch.optim.Adam(params, option['lr_dis'], betas=option['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['spv_decay_epoch'], gamma=option['decay_rate'])

    return optimizer, scheduler

def sample_p_0(n=option['batch_size'], sig=option['e_init_sig']):
    return sig * torch.randn(*[n, option['latent_dim'], 12, 12]).to(device)




def train_one_epoch(writer, swin_model, optimizer, train_loader, un_train_loader, loss_fun):
    swin_model.train()
    swin_loss_record = AvgMeter()
    print('Swin_Model Learning Rate: {:.2e}'.format(optimizer.param_groups[0]['lr']))
            
    
    #  1. Training with the Labeled Datasets         
    progress_bar_labeled = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['spv_epoch']))
    for i, pack in enumerate(progress_bar_labeled):
        for rate in size_rates:
            optimizer.zero_grad()
            
            images, gts, edges = pack[0].cuda(), pack[1].cuda(), pack[2].cuda()
            gts_contour = gts * edges
            gts_body = gts - gts * edges

            pred_swin_body, pred_swin_contour, pred_swin_pred = swin_model(images, tag="spv")

            loss_swin_body = F.binary_cross_entropy_with_logits(pred_swin_body, gts_body)
            loss_swin_contour = F.binary_cross_entropy_with_logits(pred_swin_contour, gts_contour)
            loss_swin_pred = cal_loss(pred_swin_pred, gts, loss_fun)

            loss_swin = loss_swin_body.mean() + loss_swin_contour.mean() + loss_swin_pred
            loss_swin.backward()
            optimizer.step()
            
            visualize_all(torch.sigmoid(pred_swin_body), gts_body, option['spv_log_path'])
            visualize_all(torch.sigmoid(pred_swin_contour), gts_contour, option['spv_log_path'])
            visualize_all(torch.sigmoid(pred_swin_pred), gts, option['spv_log_path'])
            
            if rate == 1:
                swin_loss_record.update(loss_swin.data, option['batch_size'])
        progress_bar_labeled.set_postfix(swin_loss=f'{swin_loss_record.show():.5f}')


    adjust_uspv_lr(optimizer, option['lr'], epoch, option['decay_rate'], option['spv_decay_epoch'])

    
    return swin_model, swin_loss_record

if __name__ == "__main__":
    # Begin the training process
    set_seed(option['seed'])
    loss_fun = get_loss(option)
    swin_model, _ = get_model(option)
    optimizer, scheduler = get_optim_swin(option, swin_model.parameters())
    
    task_name = option['task']
    
    train_loader, un_train_loader = get_loader(image_root=option['image_root'], gt_root=option['gt_root'], edge_root=option['edge_root'],
        ratio=option['partial_ratio'], batchsize=option['batch_size'], trainsize=option['trainsize'], patchsize=option["patchsize"])
    
    size_rates = option['size_rates']  # multi-scale training
    writer = SummaryWriter(option['spv_log_save_path'])
    for epoch in range(1, (option['spv_epoch']+1)):
        swin_model, swin_loss_record = train_one_epoch(writer, swin_model, optimizer, train_loader, un_train_loader, loss_fun)
        
        writer.add_scalar('Swin_model loss', swin_loss_record.show(), epoch)
        writer.add_scalar('Swin_model_lr', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()
        
      
        save_path = option['ckpt_save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % int(option['spv_save_epoch']) == 0:
            torch.save(swin_model.state_dict(), save_path + '/{:d}'.format(epoch) + '_spv_model.pth')
      
      
