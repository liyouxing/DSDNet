import torch
from skimage.metrics import structural_similarity

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


def single_ssim_ts3(pred, gt, data_range=1.0):
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


def batch_ssim_ts4(pred, gt, data_range=1.):
    """ average ssim in a batch"""
    pred_list = torch.split(pred, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_list_np = [pred_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze()
                    for ind in range(len(pred_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze()
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


if __name__ == "__main__":
    pass
