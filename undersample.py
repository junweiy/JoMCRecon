import numpy as np
from numpy.lib.stride_tricks import as_strided
import itertools
from sigpy.mri import poisson

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, rate, sample_n=6, centred=True):
    assert rate > 1
    if len(shape) == 4:
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    elif len(shape) == 3:
        N, Nx, Ny = shape
    else:
        raise ValueError("Invalid dimension, {} given.".format(len(shape)))

    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * rate)
    n_lines = int(Nx / rate)
    assert n_lines >= sample_n
    pdf_x += lmda * 1. / Nx
    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask


def center_square(shape, rate):
    assert rate > 1
    if len(shape) == 4:
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    elif len(shape) == 3:
        N, Nx, Ny = shape
    else:
        raise ValueError("Invalid dimension, {} given.".format(len(shape)))
    mask = np.zeros((N, Nx, Ny))
    mid_x = int(Nx / 2)
    interval_x = int(np.around(Nx / np.sqrt(rate) / 2))
    mid_y = int(Ny / 2)
    interval_y = int(np.around(Ny / np.sqrt(rate) / 2))
    mask[:, mid_x - interval_x: mid_x + interval_x, mid_y - interval_y:mid_y + interval_y] = 1
    return mask


def get_vd_mask(shape, rate, seed=0):
    if len(shape) == 2:
        W, H = shape
    else:
        raise ValueError("Invalid dimension, {} given.".format(len(shape)))
    print(W*H, rate)
    mask = poisson((W, H), rate, dtype=np.int, crop_corner=False)
    return mask


def generate_masks(dataset_name, dim, un_rates, plane):
    for un_rate in un_rates:
        square_mask = center_square((1, dim[0], dim[1]), un_rate)[0,]
        cart_mask = cartesian_mask((1, dim[0], dim[1]), un_rate)[0,]
        vd_mask = get_vd_mask((dim[0], dim[1]), un_rate).astype(int)

        np.save('./ours_mask/{dataset_name}_'.format(dataset_name=dataset_name) + str(
            1 / un_rate) + '_square_' + plane + '.npy', square_mask[..., np.newaxis])
        np.save('./ours_mask/{dataset_name}_'.format(dataset_name=dataset_name) + str(
            1 / un_rate) + '_1dgau_' + plane + '.npy', cart_mask[..., np.newaxis])
        np.save('./ours_mask/{dataset_name}_'.format(dataset_name=dataset_name) + str(
            1 / un_rate) + '_vd_' + plane + '.npy', vd_mask[..., np.newaxis])



def get_full_mask(ds_name, c_fold, un_rate, m_name, generated_mask_path, query_modal, plane='axial', aug=False):
    if m_name == 'oursloupe':
        path = generated_mask_path + ds_name + '_' + str(un_rate) + '_' + str(c_fold) + '_un_clipped_' + plane + '_' + query_modal + ('_aug' if aug else '') + '.npy'
    elif m_name == 'ours':
        path = generated_mask_path + ds_name + '_' + str(un_rate) + '_' + str(c_fold) + '_' + plane + '.npy'
    elif m_name in ['vd', 'square', '1dgau']:
        path = generated_mask_path + ds_name + '_' + str(un_rate) + '_' + m_name + '_' + plane + '.npy'
    elif m_name == 'jmodl':
        path = generated_mask_path + ds_name + '_' + str(un_rate) + '_' + query_modal + '.npy'
    elif m_name == 'loupe':
        return None
    elif m_name == 'other':
        path = generated_mask_path + 'BraTS' + '_' + str(un_rate) + '_oursloupe_' + query_modal + '.npy'
    elif m_name == 'otherloupe':
        path = generated_mask_path + 'BraTS' + '_' + str(un_rate) + '_loupe_' + query_modal + '.npy'
    else:
        raise NotImplementedError
    return np.load(path)