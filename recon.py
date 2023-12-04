import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from signal_utils import fft, ifft, mkdir

import random

from skimage.metrics import structural_similarity
from tensorboardX import SummaryWriter

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape

    max_intensity = 1
    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)



def get_ckpt_name(file_suffix, epoch, model_path):
    mkdir(model_path + "unet_%s/" % file_suffix)
    return model_path + "unet_%s/unet_%s_%d.pth" % (file_suffix, file_suffix, epoch)



def get_concated_tensor(trans_model, imgs_t1, imgs_t2u, u_mask, mode='recon'):
    if mode == 'trans':
        concated = imgs_t1
    elif mode == 'recon':
        if u_mask is not None:
            if trans_model is not None:
                exit(1)
                trans_t2 = trans_model(imgs_t1, None).clone().detach()
                concated = torch.cat((trans_t2, imgs_t2u), 1)
            else:
                concated = torch.cat((imgs_t1, imgs_t2u), 1)
        else:
            concated = torch.cat((imgs_t1, torch.zeros_like(imgs_t1)), 1)
    else:
        exit(1)
    return concated


def pred_t2_img(model, concated, u_mask, imgs_t1, mode):
    # with torch.no_grad():
    if u_mask is not None:
        gen_t2 = model(concated, torch.tensor(u_mask).cuda())
    else:
        if mode == 'recon':
            gen_t2 = model(concated, None)
        elif mode == 'trans':
            gen_t2 = model(imgs_t1, None)
    if imgs_t1.shape[1] == 3:
        t1_content_mask = (imgs_t1 != 0)
        gen_t2 *= torch.logical_or(torch.logical_or(t1_content_mask[:,:1,...], t1_content_mask[:,1:2,...]), t1_content_mask[:,2:,...])
    elif imgs_t1.shape[1] == 1:
        gen_t2 *= (imgs_t1 != 0)
    return gen_t2


def train_recon(model, trans_model, dataloader, val_dataloader, epoch, n_epochs, file_suffix, u_mask, model_path, mode='recon'):
    train_writer = SummaryWriter(comment=file_suffix + '_train')

    # Losses
    criterion_unet = nn.MSELoss()

    if cuda:
        model = model.cuda()
        criterion_unet = criterion_unet.cuda()

    optimizer_unet = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    if epoch != 0:
        model.load_state_dict(torch.load(get_ckpt_name(file_suffix, epoch - 1, model_path)))

    step = epoch * len(dataloader)
    model.train()
    for epoch in range(epoch, epoch + n_epochs):

        for i, imgs in enumerate(dataloader):

            imgs_t1, imgs_t2, imgs_t2u = get_data_pair(imgs, u_mask)
            concated = get_concated_tensor(trans_model, imgs_t1, imgs_t2u, u_mask, mode)

            model.zero_grad()
            gen_t2 = pred_t2_img(model, concated, u_mask, imgs_t1, mode)
            loss_unet = criterion_unet(imgs_t2, gen_t2)
            loss_unet.backward()
            optimizer_unet.step()

            if u_mask is not None:
                t2u_psnr = psnr(imgs_t2.cpu().numpy()[0, 0], imgs_t2u.cpu().numpy()[0, 0])
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [U-Net loss: %f] [T2u PSNR: %.4f]"
                    % (epoch, n_epochs, i, len(dataloader), loss_unet.item(), t2u_psnr)
                )
            else:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [U-Net loss: %f]"
                    % (epoch, n_epochs, i, len(dataloader), loss_unet.item())
                )

            train_writer.add_scalar('loss', loss_unet.item(), global_step=step)
            step += 1

        if epoch % 5 == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), get_ckpt_name(file_suffix, epoch, model_path))





def test_recon(model, trans_model, dataloader, file_suffix, epoch, u_mask, model_path, mode='recon'):

    model.load_state_dict(torch.load(get_ckpt_name(file_suffix, epoch, model_path)))
    model.eval()

    t2u_psnr_list = []
    recon_psnr_list = []
    recon_ssim_list = []

    for i, imgs in enumerate(dataloader):
        imgs_t1, imgs_t2, imgs_t2u = get_data_pair(imgs, u_mask)
        concated = get_concated_tensor(trans_model, imgs_t1, imgs_t2u, u_mask, mode)

        pred_t2 = pred_t2_img(model, concated, u_mask, imgs_t1, mode)

        t2u_psnr, pred_psnr, pred_ssim = get_psnr_ssim(imgs_t2, pred_t2, imgs_t2u)

        t2u_psnr_list.append(t2u_psnr)
        recon_psnr_list.append(pred_psnr)
        recon_ssim_list.append(pred_ssim)

    print('PSNR for the current dataset: ' + str(np.mean(recon_psnr_list)) + str(np.std(recon_psnr_list)))
    print('SSIM for the current dataset: ' + str(np.mean(recon_ssim_list)) + str(np.std(recon_ssim_list)))

    model.load_state_dict(torch.load(get_ckpt_name(file_suffix, epoch, model_path)))
    return model

def get_psnr_ssim(imgs_t2, pred_t2, imgs_t2u):
    pred_t2_np = pred_t2.cpu().detach().numpy()[:, 0]
    imgs_t2_np = imgs_t2.cpu().detach().numpy()[:, 0]
    t2u_psnr = 0
    if imgs_t2u is not None:
        imgs_t2u_np = imgs_t2u.cpu().detach().numpy()[:, 0] * (imgs_t2_np != 0)
        t2u_psnr = psnr(imgs_t2u_np, imgs_t2_np)
    pred_psnr = psnr(pred_t2_np, imgs_t2_np)
    pred_ssim = 0

    for sl in range(pred_t2_np.shape[0]):
        pred_ssim += structural_similarity(pred_t2_np[sl], imgs_t2_np[sl])
    pred_ssim /= pred_t2_np.shape[0]
    return t2u_psnr, pred_psnr, pred_ssim

def get_data_pair(imgs, u_mask):
    if torch.is_floating_point(imgs[0]):
        imgs_t1 = Variable(imgs[0].type(Tensor))
        imgs_t2 = Variable(imgs[1].type(Tensor))
    else:
        imgs_t1 = Variable(torch.concat((imgs[0].real, imgs[0].imag), dim=1).type(Tensor))
        imgs_t2 = Variable(torch.concat((imgs[1].real, imgs[1].imag), dim=1).type(Tensor))
    imgs_t2u = None

    if u_mask is not None:
        ku = fft(imgs[1].cpu().numpy()) * u_mask
        t2u = torch.from_numpy(ifft(ku))
        imgs_t2u = Variable(t2u.cuda())
        imgs_t2u *= (imgs_t2 != 0)
        if torch.is_complex(imgs[0]):
            imgs_t2u = torch.concat((imgs_t2u.real, imgs_t2u.imag), dim=1)
    return imgs_t1, imgs_t2, imgs_t2u

