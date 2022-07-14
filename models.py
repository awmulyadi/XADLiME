import torch
import torch.nn as nn
import math

class ADPENVAE(nn.Module):
    def __init__(self, args):
        super(ADPENVAE, self).__init__()

        self.args = args
        self.hiddens = self.args.vae_hiddens

        # Encoder
        self.fc_enc_stage = nn.Linear(self.args.spectrum_dim, 8)
        self.af = nn.ReLU()
        self.fc_enc_mmse_stage = nn.Linear(10, 10)
        self.enc = nn.Sequential()
        for i in range(len(self.hiddens)-2):
            self.enc.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i+1]))
            self.enc.add_module("af_%d" % i, nn.ReLU())
        self.enc_mu = nn.Linear(self.hiddens[-2], self.hiddens[-1])
        self.enc_logvar = nn.Linear(self.hiddens[-2], self.hiddens[-1])

        # Decoder
        self.dec = nn.Sequential()
        for i in range(len(self.hiddens))[::-1][:-2]:
            self.dec.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i-1]))
            self.dec.add_module("af_%d" % i, nn.ReLU())
        self.fc_dec_stage = nn.Linear(self.hiddens[1], self.args.spectrum_dim)
        self.fc_dec_mmse = nn.Linear(self.hiddens[1], 1)
        self.fc_dec_age = nn.Linear(self.hiddens[1], 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Intialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    # Reparameterize
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = 1e-6 + mu + eps * std
        return z

    def forward(self, x_stage, x_mmse, x_age):

        # Pre encoding
        e_stage = self.af(self.fc_enc_stage(x_stage)) # Clinical Labels
        e_all = self.af(self.fc_enc_mmse_stage(torch.cat([e_stage, x_mmse, x_age], dim=1)))

        # Encoding
        e = self.enc(e_all)
        z_mu = self.enc_mu(e)
        z_logvar =self.enc_logvar(e)
        z = self.reparameterize(z_mu, z_logvar)

        # Decoding
        d = self.dec(z)
        x_hat_stage = self.softmax(self.fc_dec_stage(d)) # Clinical Labels
        x_hat_mmse = self.fc_dec_mmse(d) # MMSE
        x_hat_age = self.fc_dec_age(d) # Age

        return z, z_mu, z_logvar, x_hat_stage, x_hat_mmse, x_hat_age

# ADPENSOM
class ADPENSOM(nn.Module):
    def __init__(self, args):
        super(ADPENSOM, self).__init__()

        # Save variables
        self.args = args
        self.l = self.args.som_map_size[0]
        self.m = self.args.som_map_size[1]
        self.n = self.args.som_map_size[2]
        self.Tmax = self.args.som_Tmax
        self.Tmin = self.args.som_Tmin
        self.T = self.Tmax
        self.n_prototypes = self.l * self.m * self.n
        self.prototypes = nn.Parameter(torch.zeros(self.args.som_dim, self.n_prototypes))

        # Intialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def init_prototypes(self, prototypes):
        self.prototypes = nn.Parameter(prototypes)

    def get_prototypes(self):
        return self.prototypes

    def neighborhood_fn(self, d, T):
        return torch.exp(-(d**2.) / (2*T ** 2.))

    # Calculate pairwise Manhattan distances between cluster assignments and map prototypes
    # (rectangular grid topology)
    def manhattan_distance(self, y_pred):
        prototype_plane = torch.arange(self.m * self.n).to(self.args.device)
        tmp = y_pred.unsqueeze(1) % (self.m * self.n)
        d_row = torch.abs(tmp // self.n - prototype_plane // self.n).repeat(1, self.l)
        d_col = torch.abs(tmp % self.n - prototype_plane % self.n).repeat(1, self.l)
        prototype_3d = torch.arange(self.m * self.n * self.l).to(self.args.device)
        tmp = y_pred.unsqueeze(1)
        d_lvl = torch.abs(tmp // (self.m * self.n) - prototype_3d // (self.m * self.n))
        return (d_row + d_col + d_lvl).float()

    def forward(self, z, iter_current, iter_max):

        # Get dimensionality
        batch_size = z.shape[0]

        # Calculate pairwise squared euclidean distances between inputs and prototype vectors
        batch_z = z.view(batch_size, 1, -1).contiguous()
        batch_prototypes = self.prototypes.t().expand(batch_size, -1, -1).contiguous()

        # Calculate distance, find BMUs
        distances = torch.pow(torch.cdist(batch_z, batch_prototypes), 2).squeeze(1)

        # Calculate distance
        bmu = distances.argmin(dim=1)

        # Update temperature parameter
        if self.args.som_decay == 'exponential':
            if iter_current > iter_max - 1:
                iter_current = iter_max - 1
            self.T = self.Tmax * (self.Tmin / self.Tmax) ** (iter_current / (iter_max - 1))
        else:
            if iter_current > iter_max - 1:
                iter_current = iter_max - 1
            self.T = self.Tmax - (self.Tmax - self.Tmin) * (iter_current / (iter_max - 1))

        # Compute topographic weights batches
        w = self.neighborhood_fn(self.manhattan_distance(bmu), self.T)

        return w.detach(), distances

class ProgAE(nn.Module):
    def __init__(self, args):
        super(ProgAE, self).__init__()

        self.args = args

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(args.som_map_size[0], args.som_map_size[0]*64, kernel_size=(args.som_map_size[1], 5), stride=1),
            nn.BatchNorm2d(args.som_map_size[0]*64),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(args.som_map_size[0]*64, args.som_map_size[0]*64, kernel_size=(args.som_map_size[1], 5), stride=1),
            nn.BatchNorm2d(args.som_map_size[0]*64),
            nn.ReLU(),
            nn.Conv2d(args.som_map_size[0]*64, args.som_map_size[0], kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Intialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, rho):
        # Encoding
        z_rho = self.enc(rho)

        # Decoding
        rho_hat = self.dec(z_rho)

        return z_rho, rho_hat

# XADLiMESynth
class XADLiMESynth(nn.Module):
    def __init__(self, args):
        super(XADLiMESynth, self).__init__()

        # Save variables
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.ch = 32

        # Encoder
        self.enc_1conv1 = self.conv3_bn_relu('enc_1conv1', 1, self.ch, k=3, s=2, p=0)
        self.enc_1conv2 = self.conv3_bn_relu('enc_1conv2', self.ch, self.ch, p=1)

        self.enc_2conv1 = self.conv3_bn_relu('enc_2conv1', self.ch, self.ch*2, k=3, s=2, p=0)
        self.enc_2conv2 = self.conv3_bn_relu('enc_2conv2', self.ch*2, self.ch*2, p=1)

        self.enc_3conv1 = self.conv3_bn_relu('enc_3conv1', self.ch*2, self.ch*4, k=3, s=2, p=0)
        self.enc_3conv2 = self.conv3_bn_relu('enc_3conv2', self.ch*4, self.ch*4, p=1)

        self.enc_4conv1 = self.conv3_bn_relu('enc_4conv1', self.ch*4, self.ch*8, k=3, s=2, p=0)
        self.enc_4conv2 = self.conv3_bn_relu('enc_4conv2', self.ch*8, self.ch*8, p=1)

        self.enc_5conv1 = self.conv3_bn_relu('enc_5conv1', self.ch*8, self.ch*16, k=(5, 6, 5), s=1, p=0)

        self.fc_synth = nn.Sequential(
            nn.Linear(self.ch*16, self.ch*8),
            nn.BatchNorm1d(self.ch*8),
            nn.ELU(),
            nn.Linear(self.ch*8, args.som_map_size[0]*args.som_map_size[1]*args.som_map_size[2]),
            nn.Sigmoid()
        )

        # Intialize Parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv3_bn_relu(self, name, ch_in, ch_out, k=3, s=1, p=1, af='ELU', do_prob=None):
        seq = nn.Sequential()
        seq.add_module(name + '_conv3d', nn.Conv3d(ch_in, ch_out, k, padding=p, stride=s))
        seq.add_module(name + '_bn', nn.BatchNorm3d(ch_out))
        # seq.add_module(name + '_bn', nn.InstanceNorm3d(ch_out))
        if af == 'ReLU':
            seq.add_module(name + '_af', nn.ReLU())
        elif af == 'LeakyReLU':
            seq.add_module(name + '_af', nn.LeakyReLU())
        elif af == 'ELU':
            seq.add_module(name + '_af', nn.ELU())

        if do_prob:
            seq.add_module(name + '_do', nn.Dropout(do_prob))
        return seq

    # Encode
    def encode(self, x):
        # x = x.unsqueeze(1)                     # B, 1, 96, 114, 96
        x = self.enc_1conv2(self.enc_1conv1(x))  # B, CH, 47, 56, 47
        x = self.enc_2conv2(self.enc_2conv1(x))  # B, CH*2, 23, 27, 23
        x = self.enc_3conv2(self.enc_3conv1(x))  # B, CH*4, 11, 13, 11
        x = self.enc_4conv2(self.enc_4conv1(x))  # B, CH*8, 5, 6, 5
        z = self.enc_5conv1(x).view(x.shape[0], -1)
        return z

    # def forward(self, x, d):
    def forward(self, x):
        z = self.encode(x)
        rho_hat = self.fc_synth(z)
        rho_hat = rho_hat.view(-1, self.args.som_map_size[0], self.args.som_map_size[1], self.args.som_map_size[2])
        return rho_hat

# XADLiMEClassifer
class XADLiMEClassifer(nn.Module):
    def __init__(self, args, n_output=2):
        super(XADLiMEClassifer, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(args.classifier_input_dim, self.args.classifier_dim),
            nn.BatchNorm1d(self.args.classifier_dim),
            nn.ELU(),
            nn.Linear(self.args.classifier_dim, n_output),
        )
        self.af = nn.Softmax()

        # Intialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x):
        y_logit = self.model(x.view(x.shape[0], -1))
        y = self.af(y_logit)
        return y_logit, y

# XADLiMERegressor
class XADLiMERegressor(nn.Module):
    def __init__(self, args, n_output=1):
        super(XADLiMERegressor, self).__init__()
        self.args = args
        self.model = nn.Sequential(
            nn.Linear(args.regressor_input_dim, self.args.regressor_dim),
            nn.BatchNorm1d(self.args.regressor_dim),
            nn.ELU(),
            nn.Linear(self.args.regressor_dim, n_output),
        )
        self.af = nn.Sigmoid()

        # Intialize Parameters
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x):
        y_logit = self.model(x.view(x.shape[0], -1))
        y = self.af(y_logit)
        return y_logit, y