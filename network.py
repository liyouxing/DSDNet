import os
import torch
import torch.nn as nn
import math
from guided_filter_pytorch.guided_filter import GuidedFilter

'function'


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


def get_residual(ts):
    max_channel = torch.max(ts, dim=1, keepdim=True)
    min_channel = torch.min(ts, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel


def decomposition(x, radius_list=None, eps_list=None):
    if radius_list is None:
        radius_list = [30]
    if eps_list is None:
        eps_list = [1]

    LF_list = []
    HF_list = []
    res = get_residual(x)
    res = res.repeat(1, 3, 1, 1)
    for radius in radius_list:
        for eps in eps_list:
            gf = GuidedFilter(radius, eps)
            LF = gf(res, x)
            LF_list.append(LF)
            HF_list.append(x - LF)
    LF = torch.cat(LF_list, dim=1)
    HF = torch.cat(HF_list, dim=1)
    return LF, HF


'network modules'


class SE(nn.Module):
    def __init__(self, inp_dim, reduction=4):
        super().__init__()
        mid_dim = int(inp_dim / reduction)
        self.avgP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp_dim, mid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mid_dim, inp_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgP(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBSE(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.convB = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0)
        )
        self.se = SE(dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        x = self.convB(x)
        out = self.act(self.se(x) + res)
        return out


class RBSELayer(nn.Module):
    def __init__(self, inp_dim, n_layer):
        super().__init__()
        self.n_layer = n_layer
        self.channel = inp_dim
        self.layers = nn.ModuleList([ResBSE(self.channel) for _ in range(self.n_layer)])

    def forward(self, x):
        for k in range(self.n_layer):
            x = self.layers[k](x)
        return x


class TLNet(nn.Module):
    def __init__(self, dim=32, n_layer=4, n_rbse=1):
        super().__init__()
        self.n_layer = n_layer

        self.conv_start = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.enc_trans = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(dim * int(math.pow(2, i)),
                          dim * int(math.pow(2, i + 1)),
                          3, 1, 1),
                nn.LeakyReLU(0.2)
            ) for i in range(self.n_layer - 1)])

        self.enc_rbse = nn.ModuleList(
            [nn.Sequential(
                RBSELayer(dim * int(math.pow(2, i)), 1)
            ) for i in range(self.n_layer - 1)])

        self.enc_down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim * int(math.pow(2, i)),
                      dim * int(math.pow(2, i)),
                      4, 2, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.latent_layer = RBSELayer(dim * int(math.pow(2, self.n_layer - 1)), n_rbse)

        self.dec_up = nn.ModuleList([nn.Sequential(
            nn.ConvTranspose2d(dim * int(math.pow(2, self.n_layer - 1 - i)),
                               dim * int(math.pow(2, self.n_layer - 2 - i)),
                               4, 2, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.dec_trans = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim * int(math.pow(2, self.n_layer - 1 - i)),
                      dim * int(math.pow(2, self.n_layer - 2 - i)),
                      3, 1, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.dec_rbse = nn.ModuleList([nn.Sequential(
            RBSELayer(dim * int(math.pow(2, self.n_layer - 2 - i)), 1),
        ) for i in range(self.n_layer - 1)])

        self.conv_end = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        cat_enc = []
        x = self.conv_start(x)
        x = self.enc_rbse[0](x)
        cat_enc.append(x)
        x = self.enc_down[0](x)

        for i in range(self.n_layer - 2):
            x = self.enc_trans[i](x)
            x = self.enc_rbse[i + 1](x)
            cat_enc.append(x)
            x = self.enc_down[i + 1](x)

        x = self.enc_trans[-1](x)
        x = self.latent_layer(x)

        for i in range(self.n_layer - 1):
            x = self.dec_up[i](x)
            x = torch.cat((x, cat_enc[self.n_layer - i - 2]), dim=1)
            x = self.dec_trans[i](x)
            x = self.dec_rbse[i](x)

        x = self.conv_end(x)
        x = x.repeat(1, 3, 1, 1)

        return x


class ALNet(nn.Module):
    def __init__(self, dim=16, n_layer=4, n_rbse=1):
        super().__init__()
        self.n_layer = n_layer

        self.conv_start = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.enc_trans = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(dim * int(math.pow(2, i)),
                          dim * int(math.pow(2, i + 1)),
                          3, 1, 1),
                nn.LeakyReLU(0.2)
            ) for i in range(self.n_layer - 1)])

        self.enc_rbse = nn.ModuleList([nn.Sequential(
            RBSELayer(dim * int(math.pow(2, i)), 1),
        ) for i in range(self.n_layer - 1)])

        self.enc_down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim * int(math.pow(2, i)),
                      dim * int(math.pow(2, i)),
                      4, 2, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.latent_layer = RBSELayer(dim * int(math.pow(2, self.n_layer - 1)), n_rbse)

        self.avgP = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(dim * int(math.pow(2, self.n_layer - 1)), dim)
        self.fc2 = nn.Linear(dim, 3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.conv_start(x)
        x = self.enc_rbse[0](x)
        x = self.enc_down[0](x)

        for i in range(self.n_layer - 2):
            x = self.enc_trans[i](x)
            x = self.enc_rbse[i + 1](x)
            x = self.enc_down[i + 1](x)

        x = self.enc_trans[-1](x)
        x = self.latent_layer(x)
        x = self.avgP(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.repeat(1, 1, h, w)

        return x


class HNet(nn.Module):
    def __init__(self, dim=64, n_layer=3, n_rbse=6):
        super().__init__()
        self.n_layer = n_layer

        self.conv_start = nn.Sequential(
            nn.Conv2d(3, dim, 7, 1, 3),
            nn.LeakyReLU(0.2)
        )

        self.enc_trans = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(dim * int(math.pow(2, i)),
                          dim * int(math.pow(2, i + 1)),
                          3, 1, 1),
                nn.LeakyReLU(0.2)
            ) for i in range(self.n_layer - 1)])

        self.enc_rbse = nn.ModuleList([nn.Sequential(
            RBSELayer(dim * int(math.pow(2, i)), 1),
        ) for i in range(self.n_layer - 1)])

        self.enc_down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim * int(math.pow(2, i)),
                      dim * int(math.pow(2, i)),
                      4, 2, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.latent_layer = RBSELayer(dim * int(math.pow(2, self.n_layer - 1)), n_rbse)

        self.dec_up = nn.ModuleList(nn.Sequential(
            nn.ConvTranspose2d(dim * int(math.pow(2, self.n_layer - i - 1)),
                               dim * int(math.pow(2, self.n_layer - i - 2)),
                               4, 2, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1))

        self.dec_trans = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim * int(math.pow(2, self.n_layer - i - 1)),
                      dim * int(math.pow(2, self.n_layer - i - 2)),
                      3, 1, 1),
            nn.LeakyReLU(0.2)
        ) for i in range(self.n_layer - 1)])

        self.dec_rbse = nn.ModuleList([nn.Sequential(
            RBSELayer(dim * int(math.pow(2, self.n_layer - i - 2)), 1),
        ) for i in range(self.n_layer - 1)])

        self.conv_end = nn.Sequential(
            nn.Conv2d(dim, 3, 7, 1, 3),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        cat_enc = []
        x = self.conv_start(x)
        x = self.enc_rbse[0](x)
        cat_enc.append(x)
        x = self.enc_down[0](x)

        for i in range(self.n_layer - 2):
            x = self.enc_trans[i](x)
            x = self.enc_rbse[i + 1](x)
            cat_enc.append(x)
            x = self.enc_down[i + 1](x)

        x = self.enc_trans[-1](x)
        x = self.latent_layer(x)

        for i in range(self.n_layer - 1):
            x = self.dec_up[i](x)
            x = torch.cat((x, cat_enc[self.n_layer - i - 2]), dim=1)
            x = self.dec_trans[i](x)
            x = self.dec_rbse[i](x)

        x = self.conv_end(x)
        return x


class DSDNet(nn.Module):
    def __init__(self, radius_list, eps_list):
        super().__init__()

        self.radius_list = radius_list
        self.eps_list = eps_list

        self.TLNet = TLNet()
        self.ALNet = ALNet()
        self.HNet = HNet()

    def forward(self, x):
        x_lf, x_hf = decomposition(x, self.radius_list, self.eps_list)

        trans_lf = self.TLNet(x_lf)  # TL
        atm_lf = self.ALNet(x_lf)  # AL

        bg_lf = (x_lf - (1 - trans_lf) * atm_lf) / (trans_lf + 0.00001)  # LB

        inp_hf = x_hf

        bg_hf = self.HNet(x_hf)  # HB

        bg = bg_lf + bg_hf  # B

        bg = torch.clamp(bg, 0., 1.)

        return bg, bg_lf, bg_hf, atm_lf, trans_lf, inp_hf


if __name__ == '__main__':
    import time

    mode = 2
    net = DSDNet()
    if mode == 1:
        # real times on RTX3090
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        net = net.cuda()
        net.eval()
        total = 0.
        ts = torch.ones([1, 3, 384, 384]).cuda()
        with torch.no_grad():
            for _ in range(1000):
                torch.cuda.synchronize()
                start = time.time()
                _ = net(ts)
                torch.cuda.synchronize()
                end = time.time()
                print(end - start)
                total += (end - start)
            print("avg:" + str(total / 100.))
    elif mode == 2:
        num_params = 0
        for k, v in net.named_parameters():
            num_params += v.numel()
        print(num_params)
