import os
import os.path
import hashlib
import errno
import torch
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from torchvision import transforms as T
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

dataset_stats = {
    'CIFAR100': {
                 'size' : 32},
    'ImageNet_R': {
                 'size' : 224}, 
                }

# transformations
def get_transform(dataset='cifar100', phase='test', aug=True, resize_imnet=False):
    transform_list = []

    # get out size
    crop_size = dataset_stats[dataset]['size']

    # get mean and std
    dset_mean = (0.0,0.0,0.0)
    dset_std = (1.0,1.0,1.0)

    if phase == 'train':
        transform_list.extend([
            transforms.RandomResizedCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dset_mean, dset_std),
                            ])
    else:
        if dataset == 'DomainNet':
            transform_list.extend([
                transforms.Resize((256,256)),
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])
        else:
            transform_list.extend([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(dset_mean, dset_std),
                                ])


    return transforms.Compose(transform_list)

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)