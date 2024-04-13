import argparse
import re
import itertools
import torch
import os
import glob
import time
from datetime import datetime
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import DerainDataset1, DerainDataset2
from network import TLNet, ALNet, DSDNet, decomposition, AtmLight
from loss.ssim_loss import SSIMLoss
from utils import batch_psnr_ts4, batch_ssim_ts4, all_batch_avg_scores


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def find_last_point(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'net_params_*.tar'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall("net_params_(.*).tar", file_)
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
    print('Total number of parameters: %d' % num_params)


def print_log(log_dir, epoch, total_epochs, epoch_interval, val_psnr, val_ssim, train_loss, lr):
    print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, '
          'Val_SSIM:{5:.4f}, Train_loss:{6:.4f}, lr:{7:.6f}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        epoch_interval, epoch, total_epochs, val_psnr, val_ssim, train_loss, lr))
    #  Recording
    with open('./{}/training_log.txt'.format(log_dir), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, '
              'Val_SSIM:{5:.4f}, Train_loss:{6:.4f}, lr:{7:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch_interval, epoch, total_epochs, val_psnr, val_ssim, train_loss, lr), file=f)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialization methods provided by CycleGAN."""

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def add_common_train_args(parser, dataset_name=None):
    train_inp_txt = 'train_rain.txt'
    train_gt_txt = 'train_norain.txt'
    val_inp_txt = 'val_rain.txt'
    val_gt_txt = 'val_norain.txt'

    epoch = 720
    train_data_dir = './data/Rain100H/train/'
    val_data_dir = './data/Rain100H/val/'
    if dataset_name == "Rain100L":
        epoch = 720
        train_data_dir = './data/Rain100L/train/'
        val_data_dir = './data/Rain100L/val/'
    elif dataset_name == "SPA":
        epoch = 8
        train_data_dir = './data/SPA/train/'
        val_data_dir = './data/datasets/SPA/val/'
    elif dataset_name == "Rain-Haze":
        epoch = 600
        train_data_dir = './data/Rain_Haze/train/'
        val_data_dir = '../data/Rain_Haze/val/'
    save_epoch_interval = epoch // 160 if epoch >= 160 else 1
    t_max = epoch // 12 if epoch >= 12 else 1

    parser.add_argument('--gpu_ids', default='0,1', help='gpu for training')
    parser.add_argument('--cudnn', type=bool, default=True, help='cudnn accelerate')
    parser.add_argument('--train_data_dir', default=train_data_dir, help='training dataset dir')
    parser.add_argument('--train_inp_txt', default=train_inp_txt, help='training input txt')
    parser.add_argument('--train_gt_txt', default=train_gt_txt, help='training gt txt')
    parser.add_argument('--val_data_dir', default=val_data_dir, help='valid dataset dir')
    parser.add_argument('--val_inp_txt', default=val_inp_txt, help='valid input txt')
    parser.add_argument('--val_gt_txt', default=val_gt_txt, help='valid gt txt')
    parser.add_argument('--isTraining', type=bool, default=True, help='training status')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle in dataloader')
    parser.add_argument('--epochs', type=int, default=epoch, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight decay in the optimizer')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='min learning rate')
    parser.add_argument('--T_max', type=int, default=t_max, help='half a cosine cycle')
    parser.add_argument('--gf_radius', default=[30], help='radius of guided filtering')
    parser.add_argument('--gf_eps', default=[1], help='epsilon of guided filtering')
    parser.add_argument('--num_workers', type=int, default=4, help='data loading thread numbers')
    parser.add_argument('--print_batch_size', type=int, default=30, help='print info with batch size interval')
    parser.add_argument('--save_epoch_interval', type=int, default=save_epoch_interval,
                        help='the frequency for saving the latest model')

    return parser


'functions for training TLNet'


def get_train_args_T(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet')
    add_common_train_args(parser, dataset_name)

    log_dir = './log_train/Rain100H/T/'
    if dataset_name == "Rain100L":
        log_dir = './log_train/Rain100L/T/'
    elif dataset_name == "SPA":
        log_dir = './log_train/SPA/T/'
    elif dataset_name == "Rain-Haze":
        log_dir = './log_train/Rain_Haze/T/'

    parser.add_argument('--log_dir', default=log_dir, help='training log dir')
    parser.add_argument('--train_batch_size', type=int, default=24, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--patch_size', type=int, default=288, help='training patch size')

    args = parser.parse_args()
    return args


def train_T(net, device, loader, optimizer, loss_f, gf_radius, gf_eps):
    net.train()
    train_loss = AverageMeter()

    for batch_id, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        inp_lf, _ = decomposition(inp, gf_radius, gf_eps)
        gt_lf, _ = decomposition(gt, gf_radius, gf_eps)
        atm_p = AtmLight(inp_lf).to(device)
        tl = net(inp_lf)
        bg_lf = (inp_lf - (1 - tl) * atm_p) / (tl + 0.00001)
        loss = loss_f(gt_lf, bg_lf)  # cal loss for a batch
        train_loss.update(loss.item(), bg_lf.size(0))  # recording
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.avg  # average loss for each train epoch


def valid_T(net, device, loader, gf_radius, gf_eps):
    net.eval()
    psnr_list = []
    ssim_list = []
    # val_loss = AverageMeter()
    with torch.no_grad():
        for batch_id, (inp, gt, _) in enumerate(loader):
            inp, gt = inp.to(device), gt.to(device)
            inp_lf, _ = decomposition(inp, gf_radius, gf_eps)
            gt_lf, _ = decomposition(gt, gf_radius, gf_eps)
            atm_p = AtmLight(inp_lf)
            tl = net(inp_lf)
            bg_lf = (inp_lf - (1 - tl) * atm_p) / (tl + 0.00001)

            # cal performance
            bs_psnr = batch_psnr_ts4(bg_lf, gt_lf)
            psnr_list.append(bs_psnr)
            bs_ssim = batch_ssim_ts4(bg_lf, gt_lf)
            ssim_list.append(bs_ssim)

    avr_psnr = all_batch_avg_scores(psnr_list)
    avr_ssim = all_batch_avg_scores(ssim_list)
    return avr_psnr, avr_ssim


def setup_and_train_T(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-params for training ---\nlearning rate: {}\ntrain batch size: {}\n'.format(
        args.lr, args.train_batch_size))

    # training log
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    # dataset
    train_dataset = DerainDataset1(crop_size=args.patch_size, data_dir=args.train_data_dir,
                                   txt_files=[args.train_inp_txt, args.train_gt_txt], isTraining=args.isTraining)
    valid_dataset = DerainDataset1(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # network
    net = TLNet().to(device)
    net = torch.nn.DataParallel(net)
    print_network(net, args.log_dir)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    # loss
    ssim_loss = SSIMLoss().to(device)

    # loading the latest model
    latest_path = os.path.join(args.log_dir, "net_params_latest.tar")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        try:
            net.module.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        except FileNotFoundError:
            print("FileNotFoundError")
        print('resuming by loading training epoch %d' % initial_epoch)
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % scheduler.get_last_lr()[0])

    else:
        initial_epoch = 0

    best_psnr = 0.
    best_ssim = 0.
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        train_loss = train_T(net, device, train_loader, optimizer, ssim_loss, args.gf_radius, args.gf_eps)
        val_psnr, val_ssim = valid_T(net, device, valid_loader, args.gf_radius, args.gf_eps)
        one_epoch_time = time.time() - time_start
        scheduler.step(epoch)

        # recording log info
        print_log(args.log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, train_loss,
                  scheduler.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': train_loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': scheduler.get_last_lr()[0]}, epoch)

        # saving the best model
        if (val_psnr >= best_psnr) and (val_ssim >= best_ssim):
            best_psnr, tmp_ssim = val_psnr, val_ssim
            save_path = os.path.join(args.log_dir, "net_params_best.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)

        # saving the latest model
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(args.log_dir, "net_params_latest.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)
    print("Finished Training")


'functions for training TLNet and ALNet'


def get_train_args_TA(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet and ALNet')
    add_common_train_args(parser, dataset_name)

    pretrained_dir = './log_train/Rain100H/T/net_params_best.tar'
    log_dir = './log_train/Rain100H/TA/'

    if dataset_name == "Rain100L":
        pretrained_dir = './log_train/Rain100L/T/net_params_best.tar'
        log_dir = './log_train/Rain100L/TA/'
    elif dataset_name == "SPA":
        pretrained_dir = './log_train/SPA/T/net_params_best.tar'
        log_dir = './log_train/SPA/TA/'
    elif dataset_name == "Rain-Haze":
        pretrained_dir = './log_train/Rain_Haze/T/net_params_best.tar'
        log_dir = './log_train/Rain_Haze/TA/'

    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--log_dir', default=log_dir, help='training log dir')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='pretrained log dir')
    parser.add_argument('--ft_lr', type=float, default=1e-6, help='the learning rate for fine-tuning')

    args = parser.parse_args()
    return args


def train_TA(netT, netA, device, loader, optimizerT, optimizerA, loss_f, gf_radius, gf_eps):
    netT.train()
    netA.train()
    train_loss = AverageMeter()

    for batch_id, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        inp_lf, _ = decomposition(inp, gf_radius, gf_eps)
        gt_lf, _ = decomposition(gt, gf_radius, gf_eps)
        tl = netT(inp_lf)
        al = netA(inp_lf)
        bg_lf = (inp_lf - (1 - tl) * al) / (tl + 0.00001)
        loss = loss_f(gt_lf, bg_lf)
        train_loss.update(loss.item(), bg_lf.size(0))  # recording
        optimizerT.zero_grad()
        optimizerA.zero_grad()
        loss.backward()
        optimizerT.step()
        optimizerA.step()

    return train_loss.avg


def valid_TA(netT, netA, device, loader, gf_radius, gf_eps):
    netT.eval()
    netA.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch_id, (inp, gt, _) in enumerate(loader):
            inp, gt = inp.to(device), gt.to(device)
            inp_lf, _ = decomposition(inp, gf_radius, gf_eps)
            gt_lf, _ = decomposition(gt, gf_radius, gf_eps)
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


def setup_and_train_TA(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-params for training ---\nlearning rate: {}\nfine tuning rate: {}\ntrain batch size: {}\n'.format(
        args.lr, args.ft_lr, args.train_batch_size))

    # training log
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    # MyDateset
    train_dataset = DerainDataset2(data_dir=args.train_data_dir, txt_files=[args.train_inp_txt, args.train_gt_txt],
                                   isTraining=args.isTraining)
    valid_dataset = DerainDataset2(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # network
    netT = TLNet().to(device)
    netA = ALNet().to(device)
    netT = torch.nn.DataParallel(netT)
    netA = torch.nn.DataParallel(netA)

    # load pretrained TLNet
    # init_weights(netA)
    checkpoint_t = torch.load(args.pretrained_dir)
    netT.module.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_t['model_state_dict'].items()})
    print_network(netT, args.log_dir)
    print_network(netA, args.log_dir)

    # optimizer & scheduler
    optimizerT = torch.optim.Adam(netT.module.parameters(), lr=args.ft_lr)
    optimizerA = torch.optim.Adam(netA.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedulerA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerA, T_max=args.T_max, eta_min=args.eta_min)

    # define the loss
    ssim_loss = SSIMLoss().to(device)

    # loading the latest model
    latest_path = os.path.join(args.log_dir, "net_params_latest.tar")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        initial_epoch = checkpoint['epoch'] + 1
        optimizerT.load_state_dict(checkpoint['optimizerT_state_dict'])
        optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
        schedulerA.load_state_dict(checkpoint['schedulerA_state_dict'])
        try:
            netT.module.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['modelT_state_dict'].items()})
            netA.module.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['modelA_state_dict'].items()})
        except FileNotFoundError:
            print("FileNotFoundError")
        print('resuming by loading training epoch %d' % initial_epoch)
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % schedulerA.get_last_lr()[0])
    else:
        initial_epoch = 0

    # training
    best_psnr = 0.
    best_ssim = 0.
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        train_loss = train_TA(netT, netA, device, train_loader, optimizerT, optimizerA, ssim_loss,
                              args.gf_radius, args.gf_eps)
        val_psnr, val_ssim = valid_TA(netT, netA, device, valid_loader, args.gf_radius, args.gf_eps)
        one_epoch_time = time.time() - time_start
        schedulerA.step(epoch)

        # recording log info
        print_log(args.log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, train_loss,
                  schedulerA.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': train_loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': schedulerA.get_last_lr()[0]}, epoch)

        # saving the best model
        if (val_psnr >= best_psnr) and (val_ssim >= best_ssim):
            best_psnr, tmp_ssim = val_psnr, val_ssim
            save_path = os.path.join(args.log_dir, "net_params_best.tar")
            torch.save({'epoch': epoch,
                        'modelT_state_dict': netT.module.state_dict(),
                        'modelA_state_dict': netA.module.state_dict(),
                        'optimizerT_state_dict': optimizerT.state_dict(),
                        'optimizerA_state_dict': optimizerA.state_dict(),
                        'schedulerA_state_dict': schedulerA.state_dict()}, save_path)

        # saving the latest model
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(args.log_dir, "net_params_latest.tar")
            torch.save({'epoch': epoch,
                        'modelT_state_dict': netT.module.state_dict(),
                        'modelA_state_dict': netA.module.state_dict(),
                        'optimizerT_state_dict': optimizerT.state_dict(),
                        'optimizerA_state_dict': optimizerA.state_dict(),
                        'schedulerA_state_dict': schedulerA.state_dict()}, save_path)
    print("Finished Training")


'functions for joint training TLNet, ALNet, and HNet (DSDNet)'


def get_train_args_TAH(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet, ALNet, and HNet')
    add_common_train_args(parser)

    pretrained_dir = './log_train/Rain100H/TA/net_params_best.tar'
    log_dir = './log_train/Rain100H/TAH/'

    if dataset_name == "Rain100L":
        pretrained_dir = './log_train/Rain100L/TA/net_params_best.tar'
        log_dir = './log_train/Rain100L/TAH/'
    elif dataset_name == "SPA":
        pretrained_dir = './log_train/SPA/TA/net_params_6.tar'
        log_dir = './log_train/SPA/TAH/'
    elif dataset_name == "Rain-Haze":
        pretrained_dir = './log_train/Rain_Haze/TA/net_params_best.tar'
        log_dir = './log_train/Rain_Haze/TAH/'

    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--pretrained_dir', default=pretrained_dir, help='pretrained log dir')
    parser.add_argument('--log_dir', default=log_dir, help='training log dir')
    parser.add_argument('--ft_lr', type=float, default=1e-6, help='the learning rate for fine-tuning')

    args = parser.parse_args()
    return args


def train_TAH(net, device, loader, optimizerTA, optimizerH, loss_f1, loss_f2, gf_radius, gf_eps):
    net.train()
    train_loss = AverageMeter()

    for batch_id, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        gt_lf, gt_hf = decomposition(gt, gf_radius, gf_eps)
        bg, bg_lf, bg_hf, _, _, _ = net(inp)
        loss = loss_f1(bg, gt) + loss_f2(bg, gt) + loss_f1(bg_hf, gt_hf) + loss_f2(bg_lf, gt_lf)
        train_loss.update(loss.item(), bg.size(0))
        optimizerTA.zero_grad()
        optimizerH.zero_grad()
        loss.backward()
        optimizerTA.step()
        optimizerH.step()

    return train_loss.avg


def valid_TAH(net, device, loader):
    net.eval()
    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for batch_id, (inp, gt, _) in enumerate(loader):
            inp, gt = inp.to(device), gt.to(device)
            bg, _, _, _, _, _ = net(inp)

            # cal average PSNR and SSIM
            bs_psnr = batch_psnr_ts4(bg, gt)
            psnr_list.append(bs_psnr)
            bs_ssim = batch_ssim_ts4(bg, gt)
            ssim_list.append(bs_ssim)

    avr_psnr = all_batch_avg_scores(psnr_list)
    avr_ssim = all_batch_avg_scores(ssim_list)
    return avr_psnr, avr_ssim


def setup_and_train_TAH(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-params for training ---\nlearning rate: {}\nfine tuning rate: {}\ntrain batch size: {}\n'.format(
        args.lr, args.ft_lr, args.train_batch_size))
    # training log
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    # MyDateset
    train_dataset = DerainDataset2(data_dir=args.train_data_dir, txt_files=[args.train_inp_txt, args.train_gt_txt],
                                   isTraining=args.isTraining)
    valid_dataset = DerainDataset2(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # network
    net = DSDNet(args.gf_radius, args.gf_eps).to(device)
    net = torch.nn.DataParallel(net)

    # load pretrained TLNet and ALNet
    checkpoint_ta = torch.load(args.pretrained_dir)
    net.module.TLNet.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint_ta['modelT_state_dict'].items()})
    net.module.ALNet.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint_ta['modelA_state_dict'].items()})
    print_network(net, args.log_dir)

    # optimizer & scheduler
    optimizerTA = torch.optim.Adam(itertools.chain(net.module.ALNet.parameters(), net.module.TLNet.parameters()),
                                   lr=args.ft_lr)
    optimizerH = torch.optim.Adam(net.module.HNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH, T_max=args.T_max, eta_min=args.eta_min)

    # loss
    ssim_loss = SSIMLoss().to(device)
    l1_loss = torch.nn.L1Loss().to(device)

    # loading the latest model
    latest_path = os.path.join(args.log_dir, "net_params_latest.tar")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        initial_epoch = checkpoint['epoch'] + 1
        optimizerTA.load_state_dict(checkpoint['optimizerTA_state_dict'])
        optimizerH.load_state_dict(checkpoint['optimizerH_state_dict'])
        schedulerH.load_state_dict(checkpoint['schedulerH_state_dict'])
        try:
            net.module.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

        except FileNotFoundError:
            print("FileNotFoundError")
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % schedulerH.get_last_lr()[0])
        print('resuming by loading training epoch %d' % initial_epoch)
    else:
        initial_epoch = 0

    # training
    best_psnr = 0.
    best_ssim = 0.
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()

        train_loss = train_TAH(net, device, train_loader, optimizerTA, optimizerH, l1_loss,
                               ssim_loss, args.gf_radius, args.gf_eps)
        val_psnr, val_ssim = valid_TAH(net, device, valid_loader)
        one_epoch_time = time.time() - time_start
        schedulerH.step(epoch)

        # recording log info
        print_log(args.log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, train_loss,
                  schedulerH.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': train_loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': schedulerH.get_last_lr()[0]}, epoch)

        # saving the best model
        if (val_psnr >= best_psnr) and (val_ssim >= best_ssim):
            best_psnr, tmp_ssim = val_psnr, val_ssim
            save_path = os.path.join(args.log_dir, "net_params_best.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizerTA_state_dict': optimizerTA.state_dict(),
                        'optimizerH_state_dict': optimizerH.state_dict(),
                        'schedulerH_state_dict': schedulerH.state_dict()}, save_path)

        # saving the latest model
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(args.log_dir, "net_params_latest.tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizerTA_state_dict': optimizerTA.state_dict(),
                        'optimizerH_state_dict': optimizerH.state_dict(),
                        'schedulerH_state_dict': schedulerH.state_dict()}, save_path)
    print("Finished Training")


if __name__ == "__main__":
    pass
