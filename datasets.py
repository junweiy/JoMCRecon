import glob
import random
import os
import numpy as np
import shutil

import torch
from torch.utils.data import Dataset

import nibabel as nib

from PIL import Image
from scipy.ndimage.interpolation import rotate

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class BraTS2019Full(Dataset):
    def __init__(self, dataset_path='./MICCAI_BraTS_2019_Data_Training/', crop_size=(192, 192),
                 volume_size=(240, 240, 155), n_slices=16, dim=2):
        self.crop_size = (192, 192, 128)[:dim] + (192, 192, 128)[dim+1:]
        self.root = str(dataset_path)
        self.full_vol_dim = volume_size
        self.file_list = []
        self.dim = dim
        # Define path and file names to save
        subvol = '_vol_' + str(volume_size[0]) + 'x' + str(volume_size[1]) + 'x' + str(volume_size[2]) + 'x' + str(
            n_slices) + '_' + str(dim)
        self.sub_vol_path = self.root + 'generated/' + subvol + '/'
        make_dirs(self.sub_vol_path)

        # Check if data already generated
        if not already_generated(self.sub_vol_path):
            # If data not generated, save data as numpy array
            list_IDsT1 = sorted(glob.glob(os.path.join(self.root, '*GG/*/*t1.nii.gz')))
            list_IDsT2 = sorted(glob.glob(os.path.join(self.root, '*GG/*/*t2.nii.gz')))
            list_IDsF = sorted(glob.glob(os.path.join(self.root, '*GG/*/*flair.nii.gz')))
            labels = sorted(glob.glob(os.path.join(self.root, '*GG/*/*_seg.nii.gz')))

            print('Brats2019, Total data:', len(list_IDsT1))
            save_local_volumes(list_IDsT1, list_IDsT2, list_IDsF, labels, n_channels=n_slices,
                                        sub_vol_path=self.sub_vol_path, crop_size=self.crop_size, dim=self.dim)
        self.file_list = extract_list(self.sub_vol_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        f_flair, f_t1, f_t2, f_label = np.array(self.file_list[index]).T
        img_t1 = np.vstack([np.load(t1) for t1 in f_t1])
        img_t2 = np.vstack([np.load(t2) for t2 in f_t2])
        img_flair = np.vstack([np.load(flair) for flair in f_flair])
        img_label = np.vstack([np.load(label) for label in f_label])
        img_query = get_query_img(self.query_modal, img_t2, img_flair)
        return img_t1, img_query, img_label


class BraTSSubset(Dataset):
    def __init__(self, file_list, query_modal, num_consecutive=3, aug=False):
        self.file_list = file_list
        self.num_consecutive = num_consecutive
        self.query_modal = query_modal
        self.aug = aug

    def __len__(self):
        return len(self.file_list) - self.num_consecutive + 1

    def __getitem__(self, index):
        f_flair, f_t1, f_t2, f_label = np.array(self.file_list[index:index + self.num_consecutive]).T
        img_t1 = np.vstack([np.load(t1) for t1 in f_t1])
        img_t2 = np.vstack([np.load(t2) for t2 in f_t2])
        img_flair = np.vstack([np.load(flair) for flair in f_flair])
        img_label = np.vstack([np.load(label) for label in f_label])
        img_query = get_query_img(self.query_modal, img_t2, img_flair)
        if self.aug:
            img_t1, img_query = augment_data(img_t1[..., np.newaxis], img_query[..., np.newaxis])
            img_t1, img_query = img_t1[...,0], img_query[1,...,0][np.newaxis,...]

        return img_t1, img_query, img_label

class MICCAIBraTS2019(Dataset):
    """
    Code for reading the infant brain MICCAIBraTS2018 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets', training_ratio=0.8, crop_size=(192, 192),
                 volume_size=(240, 240, 155), n_channels=3, num_consecutive=1, dim=2):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param split_idx: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = self.root + 'MICCAI_BraTS_2019_Data_Training/'
        self.testing_path = self.root + 'MICCAI_BraTS_2019_Data_Validation/'
        self.full_vol_dim = volume_size
        self.file_list = []
        self.full_volume = None
        self.training_ratio = training_ratio
        self.crop_size = crop_size
        self.num_consecutive = num_consecutive
        self.dim = dim
        # Define path and file names to save
        subvol = '_vol_' + str(volume_size[0]) + 'x' + str(volume_size[1]) + 'x' + str(volume_size[2]) + 'x' + str(
            n_channels)
        self.sub_vol_path = self.root + 'MICCAI_BraTS_2019_Data_Training/generated/' + mode + subvol + '/'
        make_dirs(self.sub_vol_path)

        # Check if data already generated
        if already_generated(self.sub_vol_path):
            self.file_list = extract_list(self.sub_vol_path)
            return

        # If data not generated, save data as numpy array
        list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t1.nii.gz')))
        list_IDsT2 = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*t2.nii.gz')))
        labels = sorted(glob.glob(os.path.join(self.training_path, '*GG/*/*_seg.nii.gz')))
        list_IDsT1, list_IDsT2, labels = shuffle_lists(list_IDsT1, list_IDsT2, labels, seed=17)

        split_idx = int(len(list_IDsT1) * self.training_ratio)

        if self.mode == 'train':
            print('Brats2019, Total data:', len(list_IDsT1))
            list_IDsT1 = list_IDsT1[:split_idx]
            list_IDsT2 = list_IDsT2[:split_idx]
            labels = labels[:split_idx]
            save_local_volumes(list_IDsT1, list_IDsT2, labels, n_channels=n_channels,
                                    sub_vol_path=self.sub_vol_path, crop_size=self.crop_size, dim=self.dim)

        elif self.mode == 'val':
            list_IDsT1 = list_IDsT1[split_idx:]
            list_IDsT2 = list_IDsT2[split_idx:]
            labels = labels[split_idx:]
            save_local_volumes(list_IDsT1, list_IDsT2, labels, n_channels=n_channels,
                                    sub_vol_path=self.sub_vol_path, crop_size=self.crop_size, dim=self.dim)

        elif self.mode == 'test':
            self.list_IDsT1 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t1.nii.gz')))
            self.list_IDsT2 = sorted(glob.glob(os.path.join(self.testing_path, '*GG/*/*t2.nii.gz')))
            self.labels = None
            # Todo inference code here

    def __len__(self):
        if self.mode == 'train' or self.mode == 'val':
            return len(self.file_list) - self.num_consecutive + 1
        elif self.mode == 'test':
            # TODO
            return len(self.list_IDsT1)

    def __getitem__(self, index):
        f_t1, f_t2, f_label = np.array(self.file_list[index:index + self.num_consecutive]).T
        img_t1 = np.vstack([np.load(t1) for t1 in f_t1])
        img_t2 = np.vstack([np.load(t2) for t2 in f_t2])
        img_label = np.vstack([np.load(label) for label in f_label])
        return img_t1, img_t2, img_label



def load_med_image(path, dtype):
    # Load data from path as array
    img_nii = nib.load(path)
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    # Normalize
    if dtype != 'label':
        data_max, data_min = np.max(img_np), np.min(img_np)
        img_np = (img_np - data_min) / (data_max - data_min)

    return img_np


def extract_volumes(path, img_np, modality, n_channels, crop_size, dim):
    paths = []
    w, h, d = img_np.shape
    x, y, z = np.where(img_np != 0)

    # Number of slices to be extracted left/right to the centre
    interval_l = lambda start: int(start - n_channels / 2) if n_channels % 2 == 0 else int(start - (n_channels - 1) / 2)
    interval_r = lambda start: int(start + n_channels / 2) if n_channels % 2 == 0 else int(
        start + (n_channels - 1) / 2 + 1)

    if dim == 0:
        start = int((max(x) + min(x)) / 2)
        modified_np = img_np[interval_l(start):interval_r(start), :, :]
    elif dim == 1:
        start = int((max(y) + min(y)) / 2)
        modified_np = img_np[:, interval_l(start): interval_r(start), :]
        modified_np = np.transpose(modified_np, (1, 0, 2))
    elif dim == 2:
        start = int((max(z) + min(z)) / 2)
        modified_np = img_np[:, :, interval_l(start):interval_r(start)]
        modified_np = np.transpose(modified_np, (2, 0, 1))
    else:
        raise NotImplementedError

    crop_1, crop_2 = crop_size

    for i in range(n_channels):
        curr_fname = path + str(i) + '_' + modality + '.npy'

        to_be_saved = modified_np[i,][np.newaxis,]
        if dim == 0:
            to_be_saved = to_be_saved[:, int(h / 2 - crop_1 / 2):int(h / 2 + crop_1 / 2),
                          int(d / 2 - crop_2 / 2):int(d / 2 + crop_2 / 2)]
        elif dim == 1:
            to_be_saved = to_be_saved[:, int(w / 2 - crop_1 / 2):int(w / 2 + crop_1 / 2),
                          int(d / 2 - crop_2 / 2):int(d / 2 + crop_2 / 2)]
        elif dim == 2:
            to_be_saved = to_be_saved[:, int(w / 2 - crop_1 / 2):int(w / 2 + crop_1 / 2),
                          int(h / 2 - crop_2 / 2):int(h / 2 + crop_2 / 2)]
        to_be_saved *= (to_be_saved != to_be_saved[0, 0, 0])
        np.save(curr_fname, to_be_saved)
        paths.append(curr_fname)
    return paths


def shuffle_lists(*ls, seed=777):
    l = list(zip(*ls))
    random.seed(seed)
    random.shuffle(l)
    return zip(*l)


def make_dirs(gen_path):
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)


def already_generated(gen_path):
    return len(os.listdir(gen_path)) > 0


def extract_list(gen_path):
    fnames = sorted(os.listdir(gen_path))
    lst = []
    for i in range(0, len(fnames), 4):
        lst.append((gen_path + fnames[i], gen_path + fnames[i + 1], gen_path + fnames[i + 2], gen_path + fnames[i + 3]))
    return lst


def idx2modality(x):
    if x == 0:
        return 'T1'
    elif x == 1:
        return 'T2'
    elif x == 2:
        return 'FLAIR'
    elif x == 3:
        return 'label'
    else:
        raise NotImplementedError

def save_local_volumes(*ls, n_channels, sub_vol_path, crop_size, dim):
    # Prepare variables
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    path_lst = []

    print("Volumes: ", total)
    for i in range(total):
        filename = sub_vol_path + 'id_' + str(i) + '_batch_'
        list_saved_paths = []

        for j in range(modalities - 1, -1, -1):
            t = 'label' if j == 3 else 'T1/T2/F'
            if j == 3:
                label = load_med_image(ls[j][i], t)
                img_np = label
            else:
                if len(np.unique(label)) == 2:
                    img_np = label * load_med_image(ls[j][i], t)
                else:
                    img_np = load_med_image(ls[j][i], t)
            # img_np = load_med_image(ls[j][i], t)
            split_fnames = extract_volumes(filename, img_np, idx2modality(j), n_channels, crop_size, dim=dim)
            list_saved_paths.append(split_fnames)
        list_saved_paths = [(list_saved_paths[0][k], list_saved_paths[1][k], list_saved_paths[2][k], list_saved_paths[3][k]) for k in
                            range(len(list_saved_paths[0]))]
        path_lst += list_saved_paths
    return path_lst


def k_fold_split(n_folds, file_list, curr_fold):
    assert curr_fold < n_folds
    each_fold_len = int(len(file_list) / n_folds)
    test_fold = file_list[each_fold_len * curr_fold: each_fold_len * (curr_fold + 1)]
    if curr_fold == n_folds - 1:
        validation_fold = file_list[:each_fold_len]
    else:
        validation_fold = file_list[each_fold_len * (curr_fold + 1): each_fold_len * (curr_fold + 2)]
    train_fold = [i for i in file_list if i not in validation_fold + test_fold]
    return train_fold, validation_fold, test_fold

def get_query_img(query_modal, img_t2, img_flair):
    if query_modal == 'T2':
        img_query = img_t2
    elif query_modal == 'FLAIR':
        img_query = img_flair
    else:
        raise ValueError('Query modality should be T2 or FLAIR.')
    return img_query


def augment_data(t1_gt, query_gt):
    # dim: [nb, nx, ny, nc]
    assert t1_gt.ndim == 4 and query_gt.ndim == 4
    np.random.seed(777)
    # randomly rotate t1
    rotated = random_rotate(t1_gt)
    # random translate t1
    translated = random_translate(rotated)
    return translated, query_gt


def random_rotate(img, rotate_range=(-5, 5)):
    _, h, w, _ = img.shape
    rotated_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        angle = np.random.randint(*rotate_range)
        tmp = rotate(img[i], angle, reshape=False, order=0)
        tmp = tmp[tmp.shape[0]//2-h//2:tmp.shape[0]//2+h//2, tmp.shape[1]//2-w//2:tmp.shape[1]//2+w//2,:]
        rotated_img[i] = tmp
    return rotated_img

def resize(img, size):
    return np.array(Image.fromarray(img).resize(size))

def random_translate(img, translate_range=(-5, 5)):
    _, h, w, _ = img.shape
    translated_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        interval_h = np.random.randint(*translate_range)
        interval_w = np.random.randint(*translate_range)
        tmp_img = np.zeros([h+2*abs(interval_h), w+2*abs(interval_w), 1])
        tmp_img[tmp_img.shape[0]//2-h//2+interval_h:tmp_img.shape[0]//2+h//2+interval_h,tmp_img.shape[1]//2-w//2+interval_w:tmp_img.shape[1]//2+w//2+interval_w] = img[i]
        translated_img[i] = tmp_img[tmp_img.shape[0]//2-h//2:tmp_img.shape[0]//2+h//2,tmp_img.shape[1]//2-w//2:tmp_img.shape[1]//2+w//2]
    return translated_img
