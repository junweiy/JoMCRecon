import numpy as np
from torch.utils.data import DataLoader

from datasets import k_fold_split, BraTS2019Full, BraTSSubset

from recon import train_recon, test_recon, get_data_pair, pred_t2_img, get_concated_tensor

from signal_utils import fft

from unet import ResUNet
import torch
import argparse


def k_largest_idx(a, k):
    idx = np.argpartition(-a.ravel(), k)[:k]
    idx1 = np.unravel_index(idx, a.shape)
    idx2 = np.column_stack(idx1)
    return idx1, idx2

def remove_outlier(residuals):
    assert residuals.ndim == 3
    res = np.copy(residuals)
    for i in range(residuals.shape[0]):
        q75, q25 = np.percentile(residuals[i,].ravel(), [75, 25])
        iqr = q75 - q25
        maximum = q75 + 1.5 * iqr
        res[i,][res[i,] > maximum] = maximum
    return res


def compute_residual_map(model, trans_model, dataset, u_mask=None, mode='recon'):
    # Make prediction
    res_map = np.zeros(shape=(1, DIM[0], DIM[1]))

    for i, imgs in enumerate(dataset):
        imgs_t1, imgs_t2, imgs_t2u = get_data_pair(imgs, u_mask)
        concated = get_concated_tensor(trans_model, imgs_t1, imgs_t2u, u_mask)

        model.eval()
        pred_t2 = pred_t2_img(model, concated, u_mask, imgs_t1, mode).clone().detach()

        kspace_pred = fft(pred_t2.cpu()[:, 0, ])
        kspace_true = fft(imgs_t2.cpu()[:, 0, ])
        res_map = np.vstack([res_map, kspace_pred - kspace_true])

    return res_map


def residual_map_processing(res_map, k):
    if res_map.ndim == 3:
        # Remove outliers
        res_map_wo = remove_outlier(res_map)

        # Take the average based on absolute values
        res_map_avg = np.average(abs(res_map_wo), axis=0)
    else:
        res_map_avg = res_map
    w, h = res_map_avg.shape
    idx_extract, idx_vis = k_largest_idx(res_map_avg, int(w * h * k))
    res_map_topk_vis = np.zeros_like(res_map_avg)

    for i in range(len(idx_vis)):
        res_map_topk_vis[idx_vis[i][0], idx_vis[i][1]] = res_map_avg[idx_vis[i][0], idx_vis[i][1]]

    return res_map_topk_vis


def get_undersampling_mask(k, model, dataset, u_mask=None):
    res_map = compute_residual_map(model, None, dataset, u_mask, mode='trans')
    res_map_full = residual_map_processing(res_map, 0.999)
    res_map_topk = residual_map_processing(res_map, k)
    res_map_topk_vis = res_map_topk.copy()
    res_map_topk_vis[res_map_topk_vis > 0] = 1
    return res_map_full, res_map_topk_vis


def combine_unbinarised_masks(mask1, mask2, top_k_percent):
    combined_full = residual_map_processing(mask1 + mask2, 0.999)
    combined = residual_map_processing(mask1 + mask2, top_k_percent)
    unbinarised = combined_full
    combined[combined > 0] = 1
    return unbinarised, combined




if __name__ == "__main__":
    print('start')
    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--qmodal', type=str, default='FLAIR')
    parser.add_argument('--dataset_name', type=str, default='BraTS')
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--num_consecutive', type=int, default=1, help='number of consecutive images as input.')
    parser.add_argument('--plane', type=str, default='axial')
    parser.add_argument('--nb_train_epoch', type=int, default=100)

    args = parser.parse_args()


    if args.dataset_name == 'BraTS':
        DIM = BraTS2019Full(dataset_path=args.dataset_path, dim=args.plane).crop_size
        full_filelist = BraTS2019Full(dataset_path=args.dataset_path, dim=args.plane).file_list
    else:
        raise NotImplementedError

    if args.dataset_name == 'BraTS':
        trans_unet = ResUNet(3 if args.aug else 1, 1)
        recon_unet = ResUNet(2, 1)
    else:
        raise NotImplementedError

    if cuda:
        trans_unet = trans_unet.cuda()
        recon_unet = recon_unet.cuda()


    for undersampling_rate in [4, 8]:
        print('current undersampling rate: ' + str(undersampling_rate))
        top_k = 1 / undersampling_rate

        train_list, validation_list, test_list = k_fold_split(1, full_filelist, 0)
        if args.dataset_name == 'BraTS':
            dataloader_train = DataLoader(
                BraTSSubset(train_list, args.qmodal, num_consecutive=args.num_consecutive, aug=args.aug),
                batch_size=args.batch_size,
                shuffle=True
            )

            dataloader_val = DataLoader(
                BraTSSubset(validation_list, args.qmodal, num_consecutive=args.num_consecutive, aug=args.aug),
                batch_size=args.batch_size,
            )

            dataloader_test = DataLoader(
                BraTSSubset(test_list, args.qmodal, num_consecutive=args.num_consecutive, aug=args.aug),
                batch_size=args.batch_size,
            )
        else:
            raise NotImplementedError

        print('Training the translation network')
        trans_name = args.dataset_name + '_trans_' + args.plane + '_' + args.qmodal + ('_aug' if args.aug else '')

        train_recon(trans_unet, None, dataloader_train, None, 0, args.nb_train_epoch, trans_name, None, args.model_path, mode='trans')

        if args.dataset_name == 'BraTS':
            trans_unet = test_recon(trans_unet, None, dataloader_val, trans_name, [args.nb_train_epoch], None, args.model_path, mode='trans')
        else:
            raise NotImplementedError

        print('Computing the residual map ...')
        unbinarised_res_mask, res_mask = get_undersampling_mask(top_k, trans_unet, dataloader_val)
        unbinarised_res_mask = np.clip(unbinarised_res_mask, 0, np.percentile(unbinarised_res_mask, 99))
        unbinarised_res_mask /= np.max(unbinarised_res_mask)

        print('Saving the residual map ...')
        np.save('./ours_mask/{d_name}_'.format(d_name=args.dataset_name) + str(1 / undersampling_rate) + '_' + args.plane + '_' + args.qmodal + ('_aug' if args.aug else '') + '.npy', res_mask)
        np.save('./ours_mask/{d_name}_'.format(d_name=args.dataset_name) + str(1 / undersampling_rate) + '_' + '_un_clipped_' + args.plane + '_' + args.qmodal + ('_aug' if args.aug else '') + '.npy', unbinarised_res_mask)
