# JoMCRecon
This repository provides the implementation of our paper "Fast Multi-Contrast MRI Acquisition by Optimal Sampling of Information Complementary to Pre-acquired MRI Contrast" published in IEEE TMI. The repository also includes the extension we implemented on the LOUPE algorithm for multi-contrast MRI reconstruction.


# Installation
The implementation is developed based on the following Python packages/versions, including:
```
python: 3.8.12
torch: 2.0.0
numpy: 1.20.3
tensorflow: 2.5.0
sigpy: 0.1.23
scipy: 1.10.1
```

# Running Code
The implementation includes two steps: (1) Pre-training the synthesis network to compute the residual map, and (2) Optimise the under-sampling pattern and the reconstruction network on an end-to-end basis.

## Step 1: Residual Computation
With the appropriately configured parameters in `residual_compute.py`, the script can be executed to train the synthesis network, and the residual map will be saved as `npy` file after training. An example of execution is as follows, which takes T2 as query/target contrast, and train the synthesis network for 100 epochs:
```
python residual_compute.py --qmodal T2 --nb_train_epoch 100
```

## Step 2: End-to-end Optimisation
With the obtained residual map, it can be loaded into the framework for multi-contrast MRI reconstruction. The main file is in `train_mri.py`. To train the model based on our proposed framework, the `mask_name` argument can be specified as `ours`, and the remaining parameters also need to be appropriately configured. An example of training the framework using T1w to assist with T2w reconstruction for 4-fold acceleration can be executed as:
```
python train_mri.py --mask_name ours --qmodal T2 --sparsities 0.25 --nb_train_epoch 100
```

## Training with Baselines
For training with baseline patterns (e.g., centre, VD, 1D Gaussian, etc.), the `generate_masks` function in `undersample.py` can be referred to for saving those patterns as `npy` files, and the baseline patterns can be loaded to retrospectively under-sample the images for optimisation by running `train_mri.py`. For example, the following script can be used to train on FLAIR data under-sampled using the VD pattern:
```
python train_mri.py --mask_name vd --qmodal FLAIR --sparsities 0.25 --nb_train_epoch 100
```

## Evaluation
To ensure a fair comparison, the evaluation module is implemented based on the original implementation of the LOUPE algorithm and shared among all methods in `loupe_test.py`. After training, the parameters can be configured accordingly in a similar way as above for evaluation. For example, to evaluate the reconstruction model trained using our proposed framework, the following script can be executed:
```
python loupe_test.py --mask_name ours --qmodal T2 -- sparsities 0.25 0.125 --nb_epoch 100
```

## Citation
If this implementation is helpful for your research, please consider citing our work:
```
@article{yang2022fast,
  title={Fast multi-contrast MRI acquisition by optimal sampling of information complementary to pre-acquired MRI contrast},
  author={Yang, Junwei and Li, Xiao-Xin and Liu, Feihong and Nie, Dong and Lio, Pietro and Qi, Haikun and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```
