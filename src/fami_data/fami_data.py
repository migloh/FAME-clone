from os.path import join
from torchvision.transforms import Compose, ToTensor
from fami_data.fami_dataset import FamiTrainDataset, FamiTestDataset

def transform():
    return Compose([
        ToTensor(),
    ])

def get_data(cfg, mode):
    data_dir_ms = join(mode, cfg['source_ms'])
    data_dir_pan = join(mode, cfg['source_pan'])
    data_dir_mask = join(mode,"mask")
    cfg = cfg
    return FamiTrainDataset(data_dir_ms, data_dir_pan, cfg, transform=transform(), data_dir_mask=data_dir_mask)

def get_test_data(cfg, mode):
    data_dir_ms = join(mode, cfg['test']['source_ms'])
    data_dir_pan = join(mode, cfg['test']['source_pan'])
    data_dir_mask = join(mode, "mask")
    cfg = cfg
    return FamiTestDataset(data_dir_ms, data_dir_pan, cfg, transform=transform(), data_dir_mask=data_dir_mask)