import os, shutil, torch, math
from solver.basesolver import BaseSolver
from utils.utils import make_optimizer, make_loss, save_config, save_net_config, cpsnr, cssim, no_ref_evaluate
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model.moe import Net
from torch import nn
from tensorboardX import SummaryWriter
from utils.config import save_yml

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
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:
            idx = math.pow(0.99, self.epoch - 1)
            para = 1* (idx)
            gate_cof = 1
            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                ms_image, lms_image, pan_image, mask_gt, file = batch

                if self.cuda:
                    ms_image, lms_image, pan_image, mask_gt = ms_image.to(self.device), lms_image.to(self.device), pan_image.to(self.device), mask_gt.to(self.device)
                self.optimizer.zero_grad()
                self.model.train()
                y,mask,lf_gate,hf_gate,dec_gate = self.model(lms_image, lms_image, pan_image)
                total_lowgate_loss = self.gate_loss(lf_gate)
                total_highgate_loss = self.gate_loss(hf_gate)
                total_decodergate_loss = self.gate_loss(dec_gate)
                mask_loss = self.mask_loss(mask_gt,mask)
                gl = total_lowgate_loss+total_highgate_loss+total_decodergate_loss
                loss = (self.loss(y, ms_image)+para*mask_loss+gl)
                if self.cfg['schedule']['use_YCbCr']:
                    y_vgg = torch.unsqueeze(y[:,3,:,:], 1)
                    y_vgg_3 = torch.cat([y_vgg, y_vgg, y_vgg], 1)
                    pan_image_3 = torch.cat([pan_image, pan_image, pan_image], 1)
                    vgg_loss = self.vggloss(y_vgg_3, pan_image_3)
                epoch_loss += loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
            self.scheduler.step()
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            self.writer.add_image('image1', ms_image[0], self.epoch)
            self.writer.add_image('image2', y[0], self.epoch)
            self.writer.add_image('image3', pan_image[0], self.epoch)
            save_config(self.cfg['log_dir'], self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)
            torch.cuda.empty_cache() 

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list,qnr_list,d_lambda_list,d_s_list = [], [],[],[],[]

            # torch.cuda.empty_cache() 
            for iteration, batch in enumerate(self.val_loader, 1):
                ms_image, lms_image, pan_image, bms_image, file = batch
                if self.cuda:
                    ms_image, lms_image, pan_image, bms_image = ms_image.to(self.device), lms_image.to(self.device), pan_image.to(self.device), bms_image.to(self.device)

                self.model.eval()
                with torch.no_grad():
                    y,mask,lf_gate,hf_gate,dec_gate = self.model(lms_image, bms_image, pan_image)

                    loss = self.loss(y, ms_image)

                batch_psnr, batch_ssim,batch_qnr,batch_D_lambda,batch_D_s = [], [],[],[],[]
                fake_img = y[:,:,:,:]
                for c in range(y.shape[0]):
                    if not self.cfg['data']['normalize']:
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        pan = (pan_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        l_ms = (lms_image[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        f_img = (fake_img[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:          
                        predict_y = (y[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (ms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        pan = (pan_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        l_ms = (lms_image[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        f_img  =(fake_img[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = cpsnr(predict_y, ground_truth)
                    ssim = cssim(predict_y,ground_truth,255)
                    l_ms = np.uint8(l_ms)
                    pan = np.uint8(pan)
                    c_D_lambda, c_D_s, QNR = no_ref_evaluate(f_img,pan,l_ms)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                    batch_qnr.append(QNR)
                    batch_D_s.append(c_D_s)
                    batch_D_lambda.append(c_D_lambda)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                avg_qnr = np.array(batch_qnr).mean()
                avg_d_lambda = np.array(batch_D_lambda).mean()
                avg_d_s = np.array(batch_D_s).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                qnr_list.extend(batch_qnr)
                d_s_list.extend(batch_D_s)
                d_lambda_list.extend(batch_D_lambda)
                t1.set_postfix_str('n:Batch loss: {:.4f}, PSNR: {:.4f}, SSIM: {:.4f},QNR:{:.4F} DS:{:.4f},D_L:{:.4F}'.format(loss.item(), avg_psnr, avg_ssim,avg_qnr,avg_d_s,avg_d_lambda))
                t1.update()
            self.records['Epoch'].append(self.epoch)
            self.records['PSNR'].append(np.array(psnr_list).mean())
            self.records['SSIM'].append(np.array(ssim_list).mean())
            self.records['QNR'].append(np.array(qnr_list).mean())
            self.records['D_lamda'].append(np.array(d_lambda_list).mean())
            self.records['D_s'].append(np.array(d_s_list).mean())
            save_config(self.cfg['log_dir'], self.log_name, 'Val Epoch {}: PSNR={:.4f}, SSIM={:.6f},QNR={:.4f}, DS:{:.4f},D_L:{:.4F}'.format(self.epoch, self.records['PSNR'][-1],
                                                                    self.records['SSIM'][-1],self.records['QNR'][-1],self.records['D_s'][-1],self.records['D_lamda'][-1]))
            self.writer.add_scalar('PSNR_epoch', self.records['PSNR'][-1], self.epoch)
            self.writer.add_scalar('SSIM_epoch', self.records['SSIM'][-1], self.epoch)
            self.writer.add_scalar('QNR_epoch', self.records['QNR'][-1], self.epoch)
            self.writer.add_scalar('D_s_epoch', self.records['D_s'][-1], self.epoch)
            self.writer.add_scalar('D_lamda_epoch', self.records['D_lamda'][-1], self.epoch)

    def check_gpu(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            self.device = torch.device("cuda")
            self.loss = self.loss.to(self.device)
            self.model = self.model.to(self.device)

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'],strict=False)
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self,epoch):
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
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                if self.epoch % 5 == 0:
                    self.eval()
                    self.save_checkpoint(epoch=self.epoch)
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint(epoch=self.epoch)
        save_config(self.cfg['log_dir'], self.log_name, 'Training done.')