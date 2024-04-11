import argparse
import itertools
import time
import torch
import os
from matplotlib import pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import DerainDataset1, DerainDataset2
from network import TLNet, ALNet, DSDNet, decomposition
from loss.ssim_loss import SSIMLoss
from utils import print_network, find_last_point, valid_T, valid_TA, valid_TAH, print_log, AtmLight, init_weights


def add_common_args(parser, dataset_name=None):
    epoch = 720
    train_data_dir = '../../datasets/Rain100H/train/'
    train_inp_txt = 'train_rain.txt'
    train_gt_txt = 'train_norain.txt'
    val_data_dir = '../../datasets/Rain100H/val/'
    val_inp_txt = 'val_rain.txt'
    val_gt_txt = 'val_norain.txt'
    model_save_dir = './log_train/Rain100H/'
    if dataset_name == "Rain100L":
        epoch = 720
        train_data_dir = '../../datasets/Rain100L/train/'
        train_inp_txt = 'train_rain.txt'
        train_gt_txt = 'train_norain.txt'
        val_data_dir = '../../datasets/Rain100L/val/'
        val_inp_txt = 'val_rain.txt'
        val_gt_txt = 'val_norain.txt'
        model_save_dir = './log_train/Rain100L/'
    elif dataset_name == "SPA":
        epoch = 8
        train_data_dir = '../../datasets/SPA/train/'
        train_inp_txt = 'train_rain.txt'
        train_gt_txt = 'train_norain.txt'
        val_data_dir = '../../datasets/SPA/val/'
        val_inp_txt = 'val_rain.txt'
        val_gt_txt = 'val_norain.txt'
        model_save_dir = './log_train/SPA/'
    elif dataset_name == "Rain-Haze":
        epoch = 600
        train_data_dir = '../../datasets/Rain_Haze/train/'
        train_inp_txt = 'train_rain.txt'
        train_gt_txt = 'train_norain.txt'
        val_data_dir = '../../datasets/Rain_Haze/val/'
        val_inp_txt = 'val_rain.txt'
        val_gt_txt = 'val_norain.txt'
        model_save_dir = './log_train/Rain_Haze/'
    save_epoch_interval = epoch // 160 if epoch >= 160 else 1
    t_max = epoch // 12 if epoch >= 12 else 1

    parser.add_argument('--gpu_ids', default='1,2,3', help='select gpu id for training')
    parser.add_argument('--cudnn', type=bool, default=True, help='cudnn accelerate')
    parser.add_argument('--train_data_dir', default=train_data_dir, help='training dataset dir')
    parser.add_argument('--train_inp_txt', default=train_inp_txt, help='training input txt')
    parser.add_argument('--train_gt_txt', default=train_gt_txt, help='training gt txt')
    parser.add_argument('--val_data_dir', default=val_data_dir, help='valid dataset dir')
    parser.add_argument('--val_inp_txt', default=val_inp_txt, help='valid input txt')
    parser.add_argument('--val_gt_txt', default=val_gt_txt, help='valid gt txt')
    parser.add_argument('--model_save_dir', default=model_save_dir, help='log dir of pretrained model')
    parser.add_argument('--isTraining', type=bool, default=True, help='training status')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle in dataloader')
    parser.add_argument('--epochs', type=int, default=epoch, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10, help='weight decay in the optimizer')
    parser.add_argument('--eta_min', type=float, default=10e-6, help='min learning rate')
    parser.add_argument('--T_max', type=int, default=t_max, help='half a cosine cycle')
    parser.add_argument('--gf_radius', default=[30], help='radius of guided filtering')
    parser.add_argument('--gf_eps', default=[1], help='epsilon of guided filtering')
    parser.add_argument('--num_workers', type=int, default=4, help='data loading thread numbers')
    parser.add_argument('--print_batch_size', type=int, default=30, help='print info with batch size interval')
    parser.add_argument('--save_epoch_interval', type=int, default=save_epoch_interval,
                        help='the frequency of saving model')



def get_args_T(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet')
    add_common_args(parser, dataset_name)

    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--patch_size', type=int, default=288, help='training patch size')

    args = parser.parse_args()
    return args


def pretrain_TLNet(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-parameters for training ---')
    print('learning rate: {}\ntrain batch size: {}\n'.format(args.lr, args.train_batch_size))

    # training log
    time_now = datetime.strftime(datetime.now(), "%m-%d")  # data time format %m-%d
    log_dir = os.path.join(args.model_save_dir, time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # tensorboardX writer

    # dataset
    train_dataset = DerainDataset1(crop_size=args.patch_size, data_dir=args.train_data_dir,
                                   txt_files=[args.train_inp_txt, args.train_gt_txt], isTraining=args.isTraining)
    valid_dataset = DerainDataset1(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # create the network
    net = TLNet().to(device)
    net = torch.nn.DataParallel(net)
    init_weights(net)
    print_network(net, log_dir)
    # optimizer & scheduler
    optimizer = torch.optim.Adam(net.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    # define the loss
    ssim_loss = SSIMLoss().to(device)
    # loading the latest model
    initial_epoch = find_last_point(save_dir=log_dir)
    if initial_epoch > 0:
        print('resuming by loading training epoch %d' % initial_epoch)
        checkpoint = torch.load(os.path.join(log_dir, "net_params_%d.tar" % initial_epoch))
        initial_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        try:
            net.module.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        except FileNotFoundError:
            print("FileNotFoundError")
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % scheduler.get_last_lr()[0])

    # training
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        # start training
        for batch_id, train_data in enumerate(train_loader):
            net.train()
            inp, gt = train_data
            inp = inp.to(device)
            gt = gt.to(device)
            inp_lf, _ = decomposition(inp, args.gf_radius, args.gf_eps)
            gt_lf, _ = decomposition(gt, args.gf_radius, args.gf_eps)
            atm_p = AtmLight(inp_lf).to(device)
            tl = net(inp_lf)
            bg_lf = (inp_lf - (1 - tl) * atm_p) / (tl + 0.00001)
            loss = ssim_loss(gt_lf, bg_lf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not (batch_id % args.print_batch_size):
                print('Epoch: {0}, Iteration: {1}, lr: {2:.6f}, Batch_loss: {3:.4f}'.format(
                    epoch + 1, batch_id, scheduler.get_last_lr()[0], loss))
        scheduler.step(epoch)
        one_epoch_time = time.time() - time_start
        # validation
        val_psnr, val_ssim = valid_T(net, valid_loader, device, args.gf_radius, args.gf_eps)
        # recording
        print_log(log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, loss,
                  scheduler.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': scheduler.get_last_lr()[0]}, epoch)
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(log_dir, "net_params_" + str(epoch + 1) + ".tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, save_path)
    print("Finished Training")


def get_args_TA(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet and ALNet')
    add_common_args(parser, dataset_name)

    pretrained_T_path = './pretrain_model/T_net_params_666_Rain100H.tar'
    if dataset_name == "Rain100L":
        pretrained_T_path = './pretrain_model/T_net_params_562_Rain100L.tar'
    elif dataset_name == "SPA":
        pretrained_T_path = './pretrain_model/T_net_params_5_SPA.tar'
    elif dataset_name == "Rain-Haze":
        pretrained_T_path = './pretrain_model/T_net_params_550_Rain_Haze.tar'

    parser.add_argument('--train_batch_size', type=int, default=10, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--pretrained_T_path', default=pretrained_T_path, help='pretrained TLNet model')
    parser.add_argument('--ft_lr', type=float, default=10e-6, help='the learning rate for fine-tuning')

    args = parser.parse_args()
    return args


def joint_train_TA(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-parameters for training ---')
    print('learning rate: {}\nfine tuning rate: {}\ntrain batch size: {}\n'.format(
        args.lr, args.ft_lr, args.train_batch_size))

    # training log
    time_now = datetime.strftime(datetime.now(), "%m-%d")  # data time format %m-%d
    log_dir = os.path.join(args.model_save_dir, time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # tensorboardX writer

    # MyDateset
    train_dataset = DerainDataset2(data_dir=args.train_data_dir,
                                   txt_files=[args.train_inp_txt, args.train_gt_txt], isTraining=args.isTraining)
    valid_dataset = DerainDataset2(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # create the network
    netA = ALNet().to(device)
    netT = TLNet().to(device)
    netA = torch.nn.DataParallel(netA)
    netT = torch.nn.DataParallel(netT)

    # init ALNet and load pretrained TLNet
    init_weights(netA)
    checkpoint_t = torch.load(args.pretrained_T_path)
    netT.module.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint_t['model_state_dict'].items()})
    print_network(netA, log_dir)
    print_network(netT, log_dir)

    # optimizer & scheduler
    optimizerA = torch.optim.Adam(netA.module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizerT = torch.optim.Adam(netT.module.parameters(), lr=args.ft_lr)
    schedulerA = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerA, T_max=args.T_max, eta_min=args.eta_min)

    # define the loss
    ssim_loss = SSIMLoss().to(device)

    # loading the latest model
    initial_epoch = find_last_point(save_dir=log_dir)
    if initial_epoch > 0:
        print('resuming by loading training epoch %d' % initial_epoch)
        checkpoint = torch.load(os.path.join(log_dir, "net_params_%d.tar" % initial_epoch))
        initial_epoch = checkpoint['epoch'] + 1
        optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
        optimizerT.load_state_dict(checkpoint['optimizerT_state_dict'])
        schedulerA.load_state_dict(checkpoint['schedulerA_state_dict'])
        try:
            netA.module.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['modelA_state_dict'].items()})
            netT.module.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['modelT_state_dict'].items()})

        except FileNotFoundError:
            print("FileNotFoundError")
        print('continue training ... start in %d epoch' % (initial_epoch + 1))
        print('lr == %f' % schedulerA.get_last_lr()[0])

    # training
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        # start training
        for batch_id, train_data in enumerate(train_loader):
            netA.train()
            netT.train()
            inp, gt = train_data
            inp = inp.to(device)
            gt = gt.to(device)
            inp_lf, _ = decomposition(inp, args.gf_radius, args.gf_eps)
            gt_lf, _ = decomposition(gt, args.gf_radius, args.gf_eps)
            al = netA(inp_lf)
            tl = netT(inp_lf)
            bg_lf = (inp_lf - (1 - tl) * al) / (tl + 0.00001)
            loss = ssim_loss(gt_lf, bg_lf)
            optimizerA.zero_grad()
            optimizerT.zero_grad()
            loss.backward()
            optimizerA.step()
            optimizerT.step()
            if not (batch_id % args.print_batch_size):
                print('Epoch: {0}, Iteration: {1}, lr: {2:.6f}, Batch_loss: {3:.4f}'.format(
                    epoch + 1, batch_id, schedulerA.get_last_lr()[0], loss))
        schedulerA.step(epoch)
        one_epoch_time = time.time() - time_start
        # validation
        val_psnr, val_ssim = valid_TA(netA, netT, valid_loader, device, args.gf_radius, args.gf_eps)
        # recording
        print_log(log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, loss,
                  schedulerA.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': schedulerA.get_last_lr()[0]}, epoch)
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(log_dir, "net_params_" + str(epoch + 1) + ".tar")
            torch.save({'epoch': epoch,
                        'modelA_state_dict': netA.module.state_dict(),
                        'modelT_state_dict': netT.module.state_dict(),
                        'optimizerA_state_dict': optimizerA.state_dict(),
                        'optimizerT_state_dict': optimizerT.state_dict(),
                        'schedulerA_state_dict': schedulerA.state_dict()}, save_path)
    print("Finished Training")


def get_args_TAH(dataset_name):
    parser = argparse.ArgumentParser(description='Options for training TLNet, ALNet, and HNet')
    add_common_args(parser)

    pretrained_TA_path = './pretrain_model/TA_net_params_714_Rain100H.tar'
    if dataset_name == "Rain100L":
        pretrained_TA_path = './pretrain_model/TA_net_params_698_Rain100L.tar'
    elif dataset_name == "SPA":
        pretrained_TA_path = './pretrain_model/TA_net_params_6_SPA.tar'
    elif dataset_name == "Rain-Haze":
        pretrained_TA_path = './pretrain_model/TA_net_params_595_Rain_Haze.tar'

    parser.add_argument('--train_batch_size', type=int, default=6, help='training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='valid batch size')
    parser.add_argument('--pretrained_TA_path', default=pretrained_TA_path, help='pretrained TLNet and ALNet model')
    parser.add_argument('--ft_lr', type=float, default=10e-6, help='the learning rate for fine-tuning')

    args = parser.parse_args()
    return args


def joint_train_TAH(args):
    # gpu setting
    plt.switch_backend('agg')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('--- Hyper-parameters for training ---')
    print('learning rate: {}\nfine tuning rate: {}\ntrain batch size: {}\n'.format(
        args.lr, args.ft_lr, args.train_batch_size))

    # training log
    time_now = datetime.strftime(datetime.now(), "%m-%d")  # data time format %m-%d
    log_dir = os.path.join(args.model_save_dir, time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # tensorboardX writer

    # MyDateset
    train_dataset = DerainDataset2(data_dir=args.train_data_dir,
                                   txt_files=[args.train_inp_txt, args.train_gt_txt], isTraining=args.isTraining)
    valid_dataset = DerainDataset2(data_dir=args.val_data_dir, txt_files=[args.val_inp_txt, args.val_gt_txt])
    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size,
                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.val_batch_size, shuffle=args.shuffle,
                              num_workers=args.num_workers // 2 if args.num_workers >= 2 else 1)
    # create the network
    net = DSDNet().to(device)
    net = torch.nn.DataParallel(net)

    # load pretrained TLNet and ALNet
    checkpoint_ta = torch.load(args.pretrained_TA_path)
    net.module.TLNet.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint_ta['modelT_state_dict'].items()})
    net.module.ALNet.load_state_dict(
        {k.replace('module.', ''): v for k, v in checkpoint_ta['modelA_state_dict'].items()})
    print_network(net, log_dir)

    # optimizer & scheduler
    optimizerH = torch.optim.Adam(net.module.HNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH, T_max=args.T_max, eta_min=args.eta_min)
    optimizerTA = torch.optim.Adam(itertools.chain(net.module.ALNet.parameters(), net.module.TLNet.parameters()),
                                   lr=args.ft_lr)
    # define the loss
    ssim_loss = SSIMLoss().to(device)
    l1_loss = torch.nn.L1Loss().to(device)
    # loading the latest model
    initial_epoch = find_last_point(save_dir=log_dir)
    if initial_epoch > 0:
        print('resuming by loading training epoch %d' % initial_epoch)
        checkpoint = torch.load(os.path.join(log_dir, "net_params_%d.tar" % initial_epoch))
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

    # training
    for epoch in range(initial_epoch, args.epochs):
        time_start = time.time()
        # start training
        for batch_id, train_data in enumerate(train_loader):
            net.train()
            inp, gt = train_data
            inp = inp.to(device)
            gt = gt.to(device)
            gt_lf, gt_hf = decomposition(gt, args.gf_radius, args.gf_eps)
            bg, bg_lf, bg_hf, _, _, _ = net(inp)
            loss = l1_loss(bg, gt) + ssim_loss(bg, gt) + l1_loss(bg_hf, gt_hf) + ssim_loss(bg_lf, gt_lf)
            optimizerTA.zero_grad()
            optimizerH.zero_grad()
            loss.backward()
            optimizerTA.step()
            optimizerH.step()
            if not (batch_id % args.print_batch_size):
                print('Epoch: {0}, Iteration: {1}, lr: {2:.6f}, Batch_loss: {3:.4f}'.format(
                    epoch + 1, batch_id, schedulerH.get_last_lr()[0], loss))
        schedulerH.step(epoch)
        one_epoch_time = time.time() - time_start
        # validation
        val_psnr, val_ssim = valid_TAH(net, valid_loader, device)
        # recording
        print_log(log_dir, epoch + 1, args.epochs, one_epoch_time, val_psnr, val_ssim, loss,
                  schedulerH.get_last_lr()[0])
        writer.add_scalars('train_loss', {'loss': loss}, epoch)
        writer.add_scalars('learning_rate', {'lr': schedulerH.get_last_lr()[0]}, epoch)
        if epoch % args.save_epoch_interval == 0:
            save_path = os.path.join(log_dir, "net_params_" + str(epoch + 1) + ".tar")
            torch.save({'epoch': epoch,
                        'model_state_dict': net.module.state_dict(),
                        'optimizerTA_state_dict': optimizerTA.state_dict(),
                        'optimizerH_state_dict': optimizerH.state_dict(),
                        'schedulerH_state_dict': schedulerH.state_dict()}, save_path)
    print("Finished Training")


if __name__ == "__main__":
    pass
