"""
    LOUPE training example (v2) with MR slices

    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Primary mail: cagladeniz94@gmail.com

    Please cite the below paper for the code:

    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu.
    "Learning-based Optimization of the Under-sampling Pattern in MRI."
    IPMI 2019
    arXiv preprint arXiv:1901.01960 (2019).
"""

# imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow.compat.v1 as tf
# from keras.backend.tensorflow_backend import set_session
from tensorflow.python.keras.backend import set_session
from keras.callbacks import ModelCheckpoint
import keras.models

# loupe
import loupe

from datasets import k_fold_split, BraTS2019Full, augment_data
import argparse

from keras.optimizers import adam_v2
from undersample import get_full_mask

###############################################################################
# parameters
###############################################################################

# TODO put them in the form of ArgumentParser()
#   see e.g. https://github.com/voxelmorph/voxelmorph/blob/master/src/train.py

parser = argparse.ArgumentParser(description='LOUPE Tester.')
parser.add_argument("--mask_name", type=str, default='jmodl', help="Name of the mask.")
parser.add_argument("--t2_only", type=bool, default=False, help='Use only T2 modality.')
parser.add_argument("--dataset_name", type=str, default='UII', help='Name of the dataset.')
parser.add_argument("--sparsities", type=int, default=[1/4, 1/8], nargs='+', help='Desired sparsity for undersampling.')
parser.add_argument("--plane", type=str, default='axial', help='plane to train, can be axial, coronal, or sagittal.')
parser.add_argument("--qmodal", type=str, default='T2Flair', help='query modality, can be t2 or flair.')
parser.add_argument("--aug", type=bool, default=False, help='Whether to apply rigid data augmentation to allow simulation of patient movement.')
parser.add_argument('--dataset_path', type=str, default='./dataset/')
parser.add_argument('--model_path', type=str, default='./saved_models/')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--pmask_slope', type=int, default=5, help='slope for prob mask sigmoid.')
parser.add_argument('--sample_slope', type=int, default=12, help='slope after sampling via uniform mask.')
parser.add_argument('--nb_epoch', type=int, default=100, help='number of epoch to be evaluated.')
parser.add_argument('--lr', type=float, default=0.001, help='the learning rate.')
parser.add_argument('--mask_path', type=str, default='./ours_mask/', help='path to save generated masks')
parser.add_argument('--loss', type=str, default='mae', help='loss function.')

args = parser.parse_args()


###############################################################################
# Data - 2D MRI slices
###############################################################################

# our data for this demo is stored in npz files.
# Please change this to suit your needs
print('loading data...')
if args.dataset_name == 'BraTS':
    full_filelist = BraTS2019Full(args.dataset_path, dim=args.plane).file_list
else:
    raise NotImplementedError

for desired_sparsity in args.sparsities:
    if args.dataset_name == 'BraTS':
        train_list, validation_list, test_list = k_fold_split(1, full_filelist, 0)
        x_test_data_t1 = np.stack([np.load(f[1]) for f in test_list], 0)[:, 0, ..., np.newaxis]
        if args.qmodal == 'T2':
            x_test_data_query = np.stack([np.load(f[2]) for f in test_list], 0)[:, 0, ..., np.newaxis]
        elif args.qmodal == 'FLAIR':
            x_test_data_query = np.stack([np.load(f[0]) for f in test_list], 0)[:, 0, ..., np.newaxis]
        else:
            raise ValueError('Query modality can only be T2 or FLAIR.')
        if args.aug:
            x_test_data_t1, x_test_data_query = augment_data(x_test_data_t1, x_test_data_query)
        x_test_data = np.concatenate([x_test_data_t1, x_test_data_query], 3)

        vol_size = x_test_data_t1.shape[1:-1]

    else:
        raise NotImplementedError

    # session initialisation
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    ###############################################################################
    # Prepare model
    ###############################################################################
    loupe_residual = True if args.mask_name == 'oursloupe' else False
    if args.dataset_name == 'BraTS':
        model = loupe.models.loupe_model(input_shape=vol_size + (1 if args.t2_only else 2,),
                                         filt=64,
                                         kern=3,
                                         model_type='v2',
                                         pmask_slope=args.pmask_slope,
                                         sparsity=desired_sparsity,
                                         sample_slope=args.sample_slope,
                                         umask=get_full_mask(args.dataset_name, 0, desired_sparsity, args.mask_name, args.mask_path, args.qmodal),
                                         loupe_residual=loupe_residual,
                                         dataset_name=args.dataset_name)
    else:
        raise NotImplementedError

    # compile
    model.compile(optimizer=adam_v2.Adam(lr=args.lr), loss=args.loss,
                  metrics=[loupe.models.compute_psnr, loupe.models.compute_ssim])

    # prepare save sub-folder
    local_name = '{prefix}_{loss}_{pmask_slope}_{sample_slope}_{sparsity}_{lr}_{dataset_name}_{plane}_{qmodal}{aug}_r_only'.format(
        prefix=args.mask_name + '_T2only' if args.t2_only else args.mask_name,
        loss=args.loss,
        pmask_slope=args.pmask_slope,
        sample_slope=args.sample_slope,
        sparsity=desired_sparsity,
        lr=args.lr,
        dataset_name=args.dataset_name,
        plane=args.plane,
        qmodal=args.qmodal,
        aug='_aug' if args.aug else '')
    save_dir_loupe = os.path.join(args.model_path, local_name)
    if not os.path.isdir(save_dir_loupe): os.makedirs(save_dir_loupe)
    filename = os.path.join(save_dir_loupe, 'model.{epoch:02d}.h5')

    ###############################################################################
    # Evaluate model
    ###############################################################################
    model.load_weights(filename.format(epoch=args.nb_epoch), by_name=True)
    psnrs = []
    ssims = []
    for i in range(len(x_test_data)):
        if args.dataset_name == 'BraTS':
            if args.t2_only:
                results = model.evaluate(x_test_data[i:i+1, ..., 1, np.newaxis], x_test_data[i:i+1, ..., 1, np.newaxis], verbose=0)
            else:
                results = model.evaluate(x_test_data[i:i+1], x_test_data[i:i+1, ..., 1, np.newaxis], verbose=0)
        else:
            raise NotImplementedError
        psnr = results[1]
        ssim = results[2]
        psnrs.append(psnr)
        ssims.append(ssim)

    print(local_name)
    print(np.mean(psnrs), np.std(psnrs), np.mean(ssims), np.std(ssims))

    del model
    tf.reset_default_graph()