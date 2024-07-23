from utils.config import get_config
from fami_data.fami_data import get_data
from model.moe import Net
import argparse
import torch
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option_path', type=str, default='./option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    ds = get_data(cfg, cfg['data_dir_train'])
    train_loader = DataLoader(ds, cfg['data']['batch_size'], shuffle=False, 
            num_workers=cfg['threads'])
    # ms_image, lms_image, pan_image, bms, fl = ds[0]
    # lms_image = torch.unsqueeze(lms_image, 0)
    # pan_image = torch.unsqueeze(pan_image, 0)
    model = Net(
        num_channels = cfg['data']['n_colors'],
        base_filter=32,
        args = cfg
    )
    for iteration, batch in enumerate(train_loader, 1):
        ms, lms, pan, mask_gt, file = batch
        y,mask,lf_gate,hf_gate,dec_gate = model(lms, lms, pan)
        print(y.shape, " y")
        print(mask.shape, " mask")
        print(lf_gate.shape, " lf_gate")
        print(hf_gate.shape, " lf_gate")
        print(dec_gate.shape, " dec_gate")
        if iteration==1:
            break