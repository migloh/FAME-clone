import os, time
import torch
from solver.basesolver import BaseSolver
from model.moe import Net
import torch.backends.cudnn as cudnn
from fami_data.fami_data import get_test_data
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tifffile import imwrite

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class TestSolver(BaseSolver):
    def __init__(self, cfg):
        super(TestSolver, self).__init__(cfg)

        self.model = Net(
            num_channels = self.cfg['data']['n_colors'],
            base_filter=32,
            args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda") 
        
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
            self.model_path = self.cfg['checkpoint']
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        avg_time = list()
        with tqdm(total=len(self.data_loader), miniters=1) as testertqdm:
            for iteration, batch in enumerate(self.data_loader, 1):
                ms_image, lms_image, pan_image, bms_image, name = batch
                if self.cuda:
                    ms_image = ms_image.to(self.device)
                    lms_image = lms_image.to(self.device)
                    pan_image = pan_image.to(self.device)
                    bms_image = bms_image.to(self.device)
                t0 = time.time()
                with torch.no_grad():
                    prediction,mask,lf_gate,hf_gate,dec_gate = self.model(lms_image, bms_image, pan_image)
                    
                # exit(0)
                t1 = time.time()

                if self.cfg['data']['normalize']:
                    ms_image = (ms_image + 1) / 2
                    lms_image = (lms_image + 1) / 2
                    pan_image = (pan_image + 1) / 2
                    bms_image = (bms_image + 1) / 2
                # print(mask[0][1])
                # break
                # print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
                avg_time.append(t1 - t0)
                # self.save_img(lms_image.cpu().data, name[0][0:-4] + '_lms.tif')  
                # self.save_img(ms_image.cpu().data, name[0][0:-4] + '_gt.tif')
                self.save_img(prediction.cpu().data, name[0][0:-4] + '_pred.tif')
                testertqdm.update()

    def save_img(self, img, img_name):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name
        save_img = np.uint8(img*255).astype('uint8') #
        imwrite(save_fn, save_img, photometric='rgb')
    
    def run(self):
        self.check()
        self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
        self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
            num_workers=self.cfg['threads'])
        self.test()