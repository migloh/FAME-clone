from torch.utils.data import Dataset
from torch import cat
import random
from os import listdir, path
from PIL import Image, ImageOps
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])

def load_img(filepath):
    img = Image.open(filepath)
    return img

def load_ms_img(pan_filename, pan_shape, n_colors):
    ms_fn = pan_filename.split("/")[-1]
    ms_img_root = path.join(*pan_filename.split("/")[:-2], "ms")
    final_ms = np.zeros((pan_shape[1], pan_shape[0], n_colors), dtype=np.uint8) 
    for i in range(1, 1+n_colors):
        ms_layer_name = ms_fn[:-5]+str(i)+".TIF"
        ms_layer_full = path.join(ms_img_root, ms_layer_name)
        img = Image.open(ms_layer_full)
        img_arr = np.array(img)
        final_ms[:, :, i-1] = img_arr
    img_obj = Image.fromarray(final_ms)
    return img_obj

def load_mask_img(pan_filename):
    mask_fn = pan_filename.split("/")[-1]
    mask_img_root = path.join(*pan_filename.split("/")[:-2], "mask")
    mask_dir = path.join(mask_img_root, mask_fn)
    mask = Image.open(mask_dir)
    return mask

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale # patch size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
    ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
    pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
    bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = pan_image.rotate(180)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug

class FamiTrainDataset(Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(FamiTrainDataset, self).__init__()
        self.ms_image_filenames = [path.join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [path.join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.mask_image_filenames = None
        if path.isdir(data_dir_mask):
            self.mask_image_filenames = [path.join(data_dir_mask, x) for x in listdir(data_dir_mask) if is_image_file(x)]
        data_dir_mask = path.join(*data_dir_pan.split("\\")[:-1], "mask")
        self.mask_filenames = [path.join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upscale']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        pan_image = load_img(self.pan_image_filenames[index])
        ms_image = load_ms_img(self.pan_image_filenames[index], pan_image.size, self.cfg["data"]["n_colors"])
        if self.mask_image_filenames != None:
            mask_image = load_mask_img(self.pan_image_filenames[index])
            mask_image = mask_image.crop((0, 0, mask_image.size[0] // self.upscale_factor * self.upscale_factor,
                                          mask_image.size[1] // self.upscale_factor * self.upscale_factor))
            mask_image = mask_image.convert('L')

        _, file = path.split(self.ms_image_filenames[index])

        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor,
                                  ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize(
            (int(ms_image.size[0] / self.upscale_factor), int(ms_image.size[1] / self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor,
                                    pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        if self.mask_image_filenames:
            ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, mask_image,
                                                                     self.patch_size, scale=self.upscale_factor)
        else:
            ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image,
                                                                     self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        if self.mask_image_filenames:
            hf = bms_image
            lf = 1 - hf
            bms_image = cat([hf, lf], dim=0)

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames) 

class FamiTestDataset(Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None, data_dir_mask=None):
        super(FamiTestDataset, self).__init__()
        print(data_dir_mask)
        self.ms_image_filenames = [path.join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [path.join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upscale']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
    
    def __getitem__(self, index):
        pan_image = load_img(self.pan_image_filenames[index])
        ms_image = load_ms_img(self.pan_image_filenames[index], pan_image.size, self.cfg["data"]["n_colors"])
        _, file = path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor), int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)

        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)