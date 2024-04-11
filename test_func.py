import argparse
import os
import torch
import torchvision.utils as utils
from torch.utils.data import DataLoader
from dataset import DerainDataset1, DerainDataset2, DerainDataset_real
from network import TLNet, ALNet, DSDNet, decomposition
from utils import AtmLight
from utils import single_psnr_ts3, single_ssim_ts3


def add_common_test_args(parser, dataset_name=None):
    test_inp_txt = 'test_rain.txt'
    test_gt_txt = 'test_norain.txt'

    test_data_dir = '../../datasets/Rain100H/test/'
    if dataset_name == "Rain100L":
        test_data_dir = '../../datasets/Rain100L/test/'
    elif dataset_name == "SPA":
        test_data_dir = '../../datasets/SPA/test/'
    elif dataset_name == "Rain-Haze":
        test_data_dir = '../../datasets/Rain_Haze/test/'

    parser.add_argument('--gpu_ids', default='1', help='select gpu id for testing')
    parser.add_argument('--test_data_dir', default=test_data_dir, help='testing dataset dir')
    parser.add_argument('--test_inp_txt', default=test_inp_txt, help='testing input txt')
    parser.add_argument('--test_gt_txt', default=test_gt_txt, help='testing gt txt')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--is_Training', type=bool, default=False, help='training or testing')
    parser.add_argument('--num_workers', type=int, default=2, help='data loading thread number')
    parser.add_argument('--gf_radius', default=[30], help='radius of guided filtering')
    parser.add_argument('--gf_eps', default=[1], help='epsilon of guided filtering')


def get_test_args_T(dataset_name):
    parser = argparse.ArgumentParser(description='Options for testing TLNet')
    add_common_test_args(parser, dataset_name)

    results_dir = './results/Rain100H/T/'
    pretrained_dir = 'xxxxxxxxxx.tar'
    if dataset_name == "Rain100L":
        results_dir = './results/Rain100L/T/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "SPA":
        results_dir = './results/SPA/T/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "Rain-Haze":
        results_dir = './results/Rain_Haze/T/'
        pretrained_dir = 'xxxxxxxxxx.tar'

    parser.add_argument('--results_dir', default=results_dir, help='test results dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='load *.tar pretrained model')

    args = parser.parse_args()
    return args


def test_T(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    score_txt = open(args.results_dir + 'scores.txt', "a+")
    test_dataset = DerainDataset1(data_dir=args.test_data_dir, txt_files=[args.test_inp_txt, args.test_gt_txt],
                                  isTraining=args.isTraining)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    # create the network
    net = TLNet().to(device)
    # loading pretrained params
    checkpoint = torch.load(args.pretrained_dir)
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # testing
    n = 0
    total_psnr = 0.0
    total_ssim = 0.0
    for j in range(1):
        net.eval()
        for batch_id, data in enumerate(test_loader):
            with torch.no_grad():
                inp, gt, fs_name = data
                inp = inp.to(device)
                gt = gt.to(device)
                inp_lf, _ = decomposition(inp, args.gf_radius, args.gf_eps)
                gt_lf, _ = decomposition(gt, args.gf_radius, args.gf_eps)
                atm_p = AtmLight(inp_lf)
                tl = net(inp_lf)
                bg_lf = (inp_lf - (1 - tl) * atm_p) / (tl + 0.00001)

            for ind in range(args.test_batch_size):
                f_name = fs_name[ind].split('/')[-1]
                n += 1
                print(f_name, n)
                save_bg_dir = args.results_dir + 'bg_lf/'
                save_tl_dir = args.results_dir + 'tl/'
                if not os.path.exists(save_bg_dir):
                    os.makedirs(save_bg_dir)
                if not os.path.exists(save_tl_dir):
                    os.makedirs(save_tl_dir)

                # accessing scores
                psnr = single_psnr_ts3(bg_lf[ind], gt_lf[ind])
                ssim = single_ssim_ts3(bg_lf[ind], gt_lf[ind])
                total_psnr += psnr
                total_ssim += + ssim
                print("img:{}   psnr:{:.2f}   ssim:{:.4f}".format(f_name[:], psnr, ssim))
                score_txt.write("img:{}   psnr:{:.2f}   ssim:{:.4f}\n".format(f_name[:], psnr, ssim))
                utils.save_image(bg_lf[ind], save_bg_dir + '{}'.format(f_name[:]))
                utils.save_image(tl[ind], save_tl_dir + '{}'.format(f_name[:]))

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}".format(avg_psnr, avg_ssim))
    score_txt.write("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}\n".format(avg_psnr, avg_ssim))
    score_txt.close()


def get_test_args_TA(dataset_name):
    parser = argparse.ArgumentParser(description='Options for testing TLNet and ALNet')
    add_common_test_args(parser, dataset_name)

    results_dir = './results/Rain100H/TA/'
    pretrained_dir = 'xxxxxxxxxx.tar'
    if dataset_name == "Rain100L":
        results_dir = './results/Rain100L/TA/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "SPA":
        results_dir = './results/SPA/TA/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "Rain-Haze":
        results_dir = './results/Rain_Haze/TA/'
        pretrained_dir = 'xxxxxxxxxx.tar'

    parser.add_argument('--results_dir', default=results_dir, help='test results dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='load *.tar pretrained model')

    args = parser.parse_args()
    return args


def test_TA(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    score_txt = open(args.results_dir + 'scores.txt', "a+")
    test_dataset = DerainDataset2(data_dir=args.test_data_dir, txt_files=[args.test_inp_txt, args.test_gt_txt],
                                  isTraining=args.isTraining)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    # create the network
    netT = TLNet().to(device)
    netA = ALNet().to(device)
    # loading pretrained params
    checkpoint = torch.load(args.pretrained_dir)
    netT.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['modelT_state_dict'].items()})
    netA.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['modelA_state_dict'].items()})

    # testing
    n = 0
    total_psnr = 0.0
    total_ssim = 0.0
    for j in range(1):
        netT.eval()
        netA.eval()
        for batch_id, data in enumerate(test_loader):
            with torch.no_grad():
                inp, gt, fs_name = data
                inp = inp.to(device)
                gt = gt.to(device)
                inp_lf, _ = decomposition(inp, args.gf_radius, args.gf_eps)
                gt_lf, _ = decomposition(gt, args.gf_radius, args.gf_eps)
                tl = netT(inp_lf)
                al = netA(inp_lf)
                bg_lf = (inp_lf - (1 - tl) * al) / (tl + 0.00001)

            for ind in range(args.test_batch_size):
                f_name = fs_name[ind].split('/')[-1]
                n += 1
                print(f_name, n)
                save_bg_dir = args.results_dir + 'bg_lf/'
                save_tl_dir = args.results_dir + 'tl/'
                save_al_dir = args.results_dir + 'al/'

                if not os.path.exists(save_bg_dir):
                    os.makedirs(save_bg_dir)
                if not os.path.exists(save_tl_dir):
                    os.makedirs(save_tl_dir)
                if not os.path.exists(save_al_dir):
                    os.makedirs(save_al_dir)

                # accessing scores
                psnr = single_psnr_ts3(bg_lf[ind], gt_lf[ind])
                ssim = single_ssim_ts3(bg_lf[ind], gt_lf[ind])
                total_psnr += psnr
                total_ssim += + ssim
                print("img:{}   psnr:{:.2f}   ssim:{:.4f}".format(f_name[:], psnr, ssim))
                score_txt.write("img:{}   psnr:{:.2f}   ssim:{:.4f}\n".format(f_name[:], psnr, ssim))
                utils.save_image(bg_lf[ind], save_bg_dir + '{}'.format(f_name[:]))
                utils.save_image(tl[ind], save_tl_dir + '{}'.format(f_name[:]))
                utils.save_image(al[ind], save_al_dir + '{}'.format(f_name[:]))

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}".format(avg_psnr, avg_ssim))
    score_txt.write("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}\n".format(avg_psnr, avg_ssim))
    score_txt.close()


def get_test_args_TAH(dataset_name):
    parser = argparse.ArgumentParser(description='Options for testing DSDNet')
    add_common_test_args(parser, dataset_name)

    results_dir = './results/Rain100H/TAH/'
    pretrained_dir = 'xxxxxxxxxx.tar'
    if dataset_name == "Rain100L":
        results_dir = './results/Rain100L/TAH/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "SPA":
        results_dir = './results/SPA/TAH/'
        pretrained_dir = 'xxxxxxxxxx.tar'
    elif dataset_name == "Rain-Haze":
        results_dir = './results/Rain_Haze/TAH/'
        pretrained_dir = 'xxxxxxxxxx.tar'

    parser.add_argument('--results_dir', default=results_dir, help='test results dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='load *.tar pretrained model')

    args = parser.parse_args()
    return args


def test_net(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    score_txt = open(args.results_dir + 'scores.txt', "a+")
    test_dataset = DerainDataset2(data_dir=args.test_data_dir, txt_files=[args.test_inp_txt, args.test_gt_txt],
                                  isTraining=args.isTraining)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    # create the network
    net = DSDNet().to(device)
    # loading pretrained params
    checkpoint = torch.load(args.pretrained_dir)
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # testing
    n = 0
    total_psnr = 0.0
    total_ssim = 0.0
    for j in range(1):
        net.eval()
        for batch_id, data in enumerate(test_loader):
            with torch.no_grad():
                inp, gt, fs_name = data
                inp = inp.to(device)
                gt = gt.to(device)

                bg, _, _, al, tl, _ = net(inp)

            for ind in range(args.test_batch_size):
                f_name = fs_name[ind].split('/')[-1]
                n += 1
                print(f_name, n)
                save_bg_dir = args.results_dir + 'bg/'
                save_tl_dir = args.results_dir + 'tl/'
                save_al_dir = args.results_dir + 'al/'

                if not os.path.exists(save_bg_dir):
                    os.makedirs(save_bg_dir)
                if not os.path.exists(save_tl_dir):
                    os.makedirs(save_tl_dir)
                if not os.path.exists(save_al_dir):
                    os.makedirs(save_al_dir)

                # accessing scores
                psnr = single_psnr_ts3(bg[ind], gt[ind])
                ssim = single_ssim_ts3(bg[ind], gt[ind])
                total_psnr += psnr
                total_ssim += + ssim
                print("img:{}   psnr:{:.2f}   ssim:{:.4f}".format(f_name[:], psnr, ssim))
                score_txt.write("img:{}   psnr:{:.2f}   ssim:{:.4f}\n".format(f_name[:], psnr, ssim))
                utils.save_image(bg[ind], save_bg_dir + '{}'.format(f_name[:]))
                utils.save_image(tl[ind], save_tl_dir + '{}'.format(f_name[:]))
                utils.save_image(al[ind], save_al_dir + '{}'.format(f_name[:]))

    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    print("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}".format(avg_psnr, avg_ssim))
    score_txt.write("Total info:\nMean psnr:{:.2f} Mean ssim:{:.4f}\n".format(avg_psnr, avg_ssim))
    score_txt.close()


def get_test_args_real():
    parser = argparse.ArgumentParser(description='Options for testing DSDNet in real cases')

    test_data_dir = '../../datasets/Real/test/'
    test_inp_txt = 'test_rain.txt'
    results_dir = './results/Real/'
    pretrained_dir = 'xxxxxxxxxx.tar'

    parser.add_argument('--gpu_ids', default='1', help='select gpu id for testing')
    parser.add_argument('--test_data_dir', default=test_data_dir, help='testing dataset dir')
    parser.add_argument('--test_inp_txt', default=test_inp_txt, help='testing input txt')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='data loading thread number')
    parser.add_argument('--results_dir', default=results_dir, help='test results dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='load *.tar pretrained model')

    args = parser.parse_args()
    return args


def test_net_unlabeled_real(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = DerainDataset_real(data_dir=args.test_data_dir, txt_files=[args.test_inp_txt])
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    # create the network
    net = DSDNet().to(device)
    # loading pretrained params
    checkpoint = torch.load(args.pretrained_dir)
    net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # testing
    n = 0
    for j in range(1):
        net.eval()
        for batch_id, data in enumerate(test_loader):
            with torch.no_grad():
                inp, fs_name = data
                inp = inp.to(device)

                bg, _, _, _, _, _ = net(inp)

            for ind in range(args.test_batch_size):
                f_name = fs_name[ind].split('/')[-1]
                n += 1
                print(f_name, n)
                save_bg_dir = args.results_dir + 'bg/'

                if not os.path.exists(save_bg_dir):
                    os.makedirs(save_bg_dir)
                utils.save_image(bg[ind], save_bg_dir + '{}'.format(f_name[:]))
