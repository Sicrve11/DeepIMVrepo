from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from datasets import N_ATTRS

def div(x_, y_):
    return torch.div(x_, y_ + 1e-8)

def log(x_):
    return torch.log(x_ + 1e-8)

### network parameters
k_prob = 0.7

input_dims = {
    'x_dim_set': [100, 100, 100, 100],
    'y_dim': 2,
    'y_type': 'binary',
    'z_dim': 100,

    'steps_per_batch': 500
}

network_settings = {
    'h_dim_p1': 100,
    'num_layers_p1': 2,  # view-specific
    'h_dim_p2': 100,
    'num_layers_p2': 2,  # multi-view
    'h_dim_e': 300,
    'num_layers_e': 3,
    'reg_scale': 0.,  # 1e-4,
}


# 网络的集合，可接收分别对应的输入
class NetworkList(torch.nn.Module):
    def __init__(self, nets):
        super(NetworkList, self).__init__()
        self.nets = nets

    def forward(self, xs):
        """
            Send `xs` through the networks.
            Parameters
            ----------
            xs : tuple of torch.Tensor
        """
        # if isinstance(xs, (tuple,list)):
        assert len(xs) == len(self.nets)
        outputs = tuple([net(x) for net, x in zip(self.nets, xs)])
        return outputs



# Encoder(100, 300, 2)
class Encoder(nn.Module):
    """Parametrizes q(z|x).
    """
    def __init__(self, dim_in, dim_lat, dim_out):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_out),
            nn.ReLU()
        )
        self.dim_in = dim_in
        self.dim_lat = dim_lat
        self.dim_out = dim_out

    def forward(self, x):
        x = self.net(x)
        return x


# Decoder(100, 100, 2)
class Decoder(nn.Module):
    """Parametrizes p(x|z).
    """

    def __init__(self, dim_in, dim_lat, dim_out):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_out),
            nn.Softmax(dim=1)
        )
        self.dim_in = dim_in
        self.dim_lat = dim_lat
        self.dim_out = dim_out

    def forward(self, x):
        x = self.net(x)
        predict = F.one_hot(torch.argmax(x, dim=1), num_classes = 2)

        return predict


# MultiDecoder(100, 100, 2)
class MultiDecoder(nn.Module):
    """Parametrizes p(y|z). """

    def __init__(self, dim_in, dim_lat, dim_out):
        super(MultiDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_lat),
            nn.ReLU(),
            nn.Dropout(p=k_prob),

            nn.Linear(dim_lat, dim_out),
            nn.Softmax(dim=1)
        )
        self.dim_in = dim_in
        self.dim_lat = dim_lat
        self.dim_out = dim_out

    def forward(self, x):
        x = self.net(x)
        predict = F.one_hot(torch.argmax(x, dim=1), num_classes=2)

        return predict


class DeepIMV(nn.Module):
    """DeepIMV:Variational Information Bottleneck Approach.
    """
    def __init__(self):
        super(DeepIMV, self).__init__()
        self.logvar_set = {}
        self.mu_set = {}
        self.x_dims = input_dims['x_dim_set']
        self.z_dim = input_dims['z_dim']
        self.y_dim = input_dims['y_dim']

        self.Vencoders = NetworkList(
            nn.ModuleList([  # 四个模态的encoder
                Encoder(self.x_dims[0], network_settings['h_dim_e'], 2 * self.z_dim),
                Encoder(self.x_dims[1], network_settings['h_dim_e'], 2 * self.z_dim),
                Encoder(self.x_dims[2], network_settings['h_dim_e'], 2 * self.z_dim),
                Encoder(self.x_dims[3], network_settings['h_dim_e'], 2 * self.z_dim)
            ])
        )

        self.Vpredict = NetworkList(
            nn.ModuleList([  # 四个模态的encoder
                Decoder(self.z_dim, network_settings['h_dim_p1'], self.y_dim),
                Decoder(self.z_dim, network_settings['h_dim_p1'], self.y_dim),
                Decoder(self.z_dim, network_settings['h_dim_p1'], self.y_dim),
                Decoder(self.z_dim, network_settings['h_dim_p1'], self.y_dim)
            ])
        )
        self.MultiPredict = MultiDecoder(self.z_dim, network_settings['h_dim_p1'], self.y_dim)
        # self.double()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    
    def ProductOfExperts(self, masks, mu_set, logvar_set):
        # 计算poe_logvar
        tmp = 1.0
        for m in range(len(mu_set)):
            tmp += torch.reshape(masks[:, m], [-1, 1]) * div(1., torch.exp(logvar_set[m]))
        poe_var = div(1.0, tmp)
        poe_logvar = log(poe_var)

        # 计算poe_mu
        tmp = 0.0
        for m in range(len(mu_set)):
            tmp += torch.reshape(masks[:, m], [-1, 1]) * div(1., torch.exp(logvar_set[m])) * mu_set[m]
        poe_mu = poe_var * tmp

        return poe_mu, poe_logvar

    def forward(self, features, mask):
        features = tuple(features)

        # encoder
        code1 = self.Vencoders(features)

        # 分解隐变量
        for i, co in enumerate(code1):
            self.mu_set[i] = co[:, :  self.z_dim]
            self.logvar_set[i] = co[:, self.z_dim :]

        self.mu_z, self.logvar_z = self.ProductOfExperts(mask, self.mu_set, self.logvar_set)

        # 使用重参数化技巧进行隐变量的生成
        self.z = self.reparametrize(self.mu_z, self.logvar_z)
        self.z_set = [self.reparametrize(self.mu_set[i], self.logvar_set[i]) for i in range(len(self.mu_set))]

        # reconstruct inputs based on that gaussian
        predict_multi = self.MultiPredict(self.z)
        predict_spec = self.Vpredict(self.z_set)

        return predict_multi, self.mu_z, self.logvar_z, predict_spec, self.mu_set, self.logvar_set
