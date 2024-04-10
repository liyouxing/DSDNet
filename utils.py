import math
import os
import glob
import time
import torch
import re

from datetime import datetime
from skimage.metrics import structural_similarity
from network import decomposition

'access functions'


def single_psnr_ts3(pred, gt, data_range=1.):
    """
    single psnr ↑↑

    Params
    ----------
    data_range : float
        data max range, e.g. if normalization, data_range = 1.,
        if image data, data_range = 255.,

    Returns
    ----------
    psnr : float
        Return psnr, range in [0, Inf)
    """
    pred = pred.detach()
    gt = gt.detach()
    mse = torch.mean((pred - gt) ** 2)
    signal_max = data_range

    psnr = 10 * torch.log10((signal_max ** 2) / mse)

    return psnr


def batch_psnr_ts4(pred, gt, data_range=1.):
    """ average psnr in a batch """
    pred = pred.detach()
    gt = gt.detach()
    b, _, _, _ = pred.shape

    psnr = single_psnr_ts3(pred[0], gt[0], data_range)
    for i in range(b - 1):
        psnr += single_psnr_ts3(pred[i], gt[i], data_range)

    return psnr / b


def single_ssim_ts3(pred, gt, data_range=255.0):
    """
    single ssim ↑↑

    Params
    ----------
    data_range : float
        If normalization, data range = 1.,
        If image data, data range = 255.
    Returns
    ----------
    ssim : float
        Return ssim, range in [0, 1]
    """
    pred = pred.permute(1, 2, 0).data.cpu().numpy()
    gt = gt.permute(1, 2, 0).data.cpu().numpy()
    ssim = structural_similarity(pred, gt, data_range=data_range, channel_axis=-1)

    return ssim


def batch_ssim_ts4(pred, gt, data_range=255.):
    """ average ssim in a batch"""
    pred_list = torch.split(pred, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_list_np = [pred_list[ind].permute(1, 2, 0).data.cpu().numpy()
                    for ind in range(len(pred_list))]
    gt_list_np = [gt_list[ind].permute(1, 2, 0).data.cpu().numpy()
                  for ind in range(len(pred_list))]
    ssim_list = [structural_similarity(pred_list_np[ind], gt_list_np[ind], data_range=data_range,
                                       channel_axis=-1) for ind in range(len(pred_list))]
    ssim = 1.0 * sum(ssim_list) / len(ssim_list)

    return ssim


def all_batch_avg_scores(score_list):
    """ average score of all batch """
    if len(score_list) == 0:
        return 0.
    else:
        avg_score = 1.0 * sum(score_list) / len(score_list)

        return avg_score


'training functions'


def find_last_point(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*params_*.tar'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*params_(.*).tar.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def print_network(net, log_dir):
    num_params = 0
    log_txt = open(log_dir + "/net_info.txt", "a+")

    for index, params in enumerate(net.parameters()):
        num_params += params.numel()
        log_txt.write("index: {:0>4}\tparams: {:0>7}\ttime: {}\n".format(
            index, params.numel(), datetime.now()))

    log_txt.write("total: {:0>8}\ttime:{}\n".format(num_params, datetime.now()))
    log_txt.close()

    # print(net)
    print('Total number of parameters: %d' % num_params)


def print_log(log_dir, epoch, total_epochs, epoch_interval, train_psnr, val_psnr, val_ssim, loss, lr):
    print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR:{5:.2f}, '
          'Val_SSIM:{6:.4f}, loss:{7:.4f}, lr:{8:.6f}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        epoch_interval, epoch, total_epochs, train_psnr,
        val_psnr, val_ssim, loss, lr))
    #  Recording
    with open('./{}/training_log.txt'.format(log_dir), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR:{5:.2f}, '
              'Val_SSIM:{6:.4f}, loss:{7:.4f}, lr:{8:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch_interval, epoch, total_epochs, train_psnr,
            val_psnr, val_ssim, loss, lr), file=f)


def AtmLight(tensor):
    """
    select highest top 0.1％ pixel and average these value to Atm
    """
    m, c, h, w = tensor.shape[::]

    vec_len = h * w
    numpx = int(max(math.floor(vec_len / 1000), 1))  # top 0.1 pixel number
    tensor_vec = tensor.reshape(m, -1)  # reshape to (m, c*h*w)
    tensor_vec = tensor_vec.sort(dim=-1, descending=True).values
    atm = torch.sum(tensor_vec[:, 0:numpx], dim=-1) / numpx
    atm = atm.unsqueeze(dim=-1).repeat(1, 3).unsqueeze(dim=-1).unsqueeze(dim=-1).repeat(1, 1, h, w)

    return atm


def valid_T(net, loader, device, radius, eps):
    psnr_list = []
    ssim_list = []

    net.eval()
    for batch_id, val_data in enumerate(loader):
        with torch.no_grad():
            inp, gt, _ = val_data
            inp = inp.to(device)
            gt = gt.to(device)

            inp_lf, _ = decomposition(inp, radius, eps)
            gt_lf, _ = decomposition(gt, radius, eps)

            atm_p = AtmLight(inp_lf)

            tl = net(inp_lf)
            bg_lf = (inp_lf - (1 - tl) * atm_p) / (tl + 0.00001)

        # cal average PSNR and SSIM
        bs_psnr = batch_psnr_ts4(bg_lf, gt_lf)
        psnr_list.append(bs_psnr)
        bs_ssim = batch_ssim_ts4(bg_lf, gt_lf)
        ssim_list.append(bs_ssim)

    avr_psnr = all_batch_avg_scores(psnr_list)
    avr_ssim = all_batch_avg_scores(ssim_list)
    return avr_psnr, avr_ssim


def valid_TA(netA, netT, loader, device, radius, eps):
    psnr_list = []
    ssim_list = []

    netA.eval()
    netT.eval()

    for batch_id, val_data in enumerate(loader):
        with torch.no_grad():
            inp, gt, _ = val_data
            inp = inp.to(device)
            gt = gt.to(device)

            inp_lf, _ = decomposition(inp, radius, eps)
            gt_lf, _ = decomposition(gt, radius, eps)

            tl = netT(inp_lf)
            atm = netA(inp_lf)
            bg_lf = (inp_lf - (1 - tl) * atm) / (tl + 0.00001)

        # cal average PSNR and SSIM
        bs_psnr = batch_psnr_ts4(bg_lf, gt_lf)
        psnr_list.append(bs_psnr)
        bs_ssim = batch_ssim_ts4(bg_lf, gt_lf)
        ssim_list.append(bs_ssim)

    avr_psnr = all_batch_avg_scores(psnr_list)
    avr_ssim = all_batch_avg_scores(ssim_list)
    return avr_psnr, avr_ssim


def valid_TAH(net, loader, device):
    psnr_list = []
    ssim_list = []

    net.eval()
    for batch_id, val_data in enumerate(loader):
        with torch.no_grad():
            inp, gt, _ = val_data
            inp = inp.to(device)
            gt = gt.to(device)

            bg, _, _, _, _, _ = net(inp)

        # cal average PSNR and SSIM
        bs_psnr = batch_psnr_ts4(bg, gt)
        psnr_list.append(bs_psnr)
        bs_ssim = batch_ssim_ts4(bg, gt)
        ssim_list.append(bs_ssim)

    avr_psnr = all_batch_avg_scores(psnr_list)
    avr_ssim = all_batch_avg_scores(ssim_list)
    return avr_psnr, avr_ssim


if __name__ == "__main__":
    pass
