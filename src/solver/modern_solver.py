from torch import nn
import torch
from solver.basesolver import BaseSolver
from model.moe import Net
from utils.utils import make_optimizer, make_loss, save_config, save_net_config
from tensorboardX import SummaryWriter
import os
from utils.config import save_yml
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import math

class CVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self,logits):
        cv = torch.std(logits,dim=1)/torch.mean(logits,dim=1)
        return self.loss_weight*torch.mean(cv**2)


class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']

        assert (self.cfg['data']['n_colors']==4)
        self.model = Net(
            num_channels = self.cfg['data']['n_colors'],
            base_filter=32,
            args = self.cfg
        )

        self.optimizer = make_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])
        self.gate_loss = CVLoss()
        self.mask_loss = make_loss(self.cfg['schedule']['loss'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 1000, 5e-8)
        self.log_name = self.cfg['algorithm'] + '_' + str(self.cfg['data']['upscale']) + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter(self.cfg['log_dir']+str(self.log_name))
        save_net_config(self.cfg['log_dir'], self.log_name, self.model)
        save_yml(cfg, os.path.join(self.cfg['log_dir'] + str(self.log_name), 'config.yml'))
        save_config(self.cfg['log_dir'], self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.cfg['log_dir'], self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.cfg['log_dir'], self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self):
        self.model.train()
        with tqdm(total=len(self.train_loader), 
                  miniters=1, 
                  desc=f'Training Epoch: [{self.epoch}/{self.nEpochs}]') as tqdm_object:
            idx = math.pow(0.99, self.epoch-1)
            para = 1 * idx
            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, mask_gt, _ = batch
                if self.cuda:
                    ms_image, lms_image, pan_image, mask_gt = ms_image.to(self.device), lms_image.to(self.device), pan_image.to(self.device), mask_gt.to(self.device)
                
                self.optimizer.zero_grad()
                y, mask, lf_gate, hf_gate, dec_gate = self.model(lms_image, lms_image, pan_image)
            
                total_lowgate_loss = self.gate_loss(lf_gate)
                total_highgate_loss = self.gate_loss(hf_gate)
                total_decodergate_loss = self.gate_loss(dec_gate)
                mask_loss = self.mask_loss(mask_gt,mask)

                all_total_loss = total_lowgate_loss+total_highgate_loss+total_decodergate_loss
                loss = self.loss(y, ms_image) + para * mask_loss + all_total_loss

                loss.backward()

                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()

                epoch_loss += loss.item()
                tqdm_object.set_postfix_str(f"Batch loss: {loss.item()}")
                tqdm_object.update()
            self.scheduler.step()

            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            self.writer.add_image('image1', ms_image[0], self.epoch)
            self.writer.add_image('image2', y[0], self.epoch)
            self.writer.add_image('image3', pan_image[0], self.epoch)
            save_config(self.cfg['log_dir'], self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)
    def eval(self):
        self.model.eval()
        with tqdm(total=len(self.val_loader), 
                  miniters=1,
                  desc=f'Eval Epoch: [{self.epoch}/{self.nEpochs}]') as tqdm_object, torch.no_grad():

            epoch_loss = 0

            for iteration, batch in enumerate(self.val_loader, 1):
                ms_image, lms_image, pan_image, bms_image, _ = batch
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.to(self.device), lms_image.to(self.device), pan_image.to(self.device), bms_image.to(self.device)
                y, mask, lf_gate, hf_gate, dec_gate = self.model(lms_image, bms_image, pan_image)
                loss = self.loss(y, ms_image)

                epoch_loss += loss.item()
                tqdm_object.set_postfix_str(f"Batch loss: {loss.item()}")
                tqdm_object.update()

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        if self.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda')
            self.loss = self.loss.to(self.device)
            self.model = self.model.to(self.device)


    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

        if self.cfg['save_best']:
            if self.records['SSIM'] != [] and self.records['SSIM'][-1] == np.array(self.records['SSIM']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestSSIM.pth'))
            if self.records['PSNR'] !=[] and self.records['PSNR'][-1]==np.array(self.records['PSNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestPSNR.pth'))
            if self.records['QNR'] !=[] and self.records['QNR'][-1]==np.array(self.records['QNR']).max():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestQNR.pth'))
            if self.records['D_lamda'] !=[] and self.records['D_lamda'][-1]==np.array(self.records['D_lamda']).min():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best_lamda.pth'))
            if self.records['D_s'] !=[] and self.records['D_s'][-1]==np.array(self.records['D_s']).min():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'bestD_s.pth'))

    def run(self):
        torch.manual_seed(self.cfg['seed'])
        self.check_gpu()
        self.train()
        self.eval()
        self.save_checkpoint()