import torch
import torch.nn as nn
from torch.nn import functional as F

class LSSLLoss(nn.Module):
    def __init__(self, z_dim, requires_grad=True, name="relu_loss", device='cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.name = name

        # Define Parameters
        self.weights = torch.nn.Parameter(
            torch.randn(1, 1, z_dim, dtype=torch.float, requires_grad=requires_grad).to(device))
        self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)
        nn.Module.register_parameter(self, 'd_weights', self.weights)

    def forward(self, z1, z2, margin=0):
        zn12 = (z1 - z2).norm(dim=1)
        z1 = torch.unsqueeze(z1, dim=1)
        z2 = torch.unsqueeze(z2, dim=1)

        # Normalization
        self.weights.data = self.weights.data.detach() / (self.weights.data.detach().norm() + 1e-10)

        h1 = F.conv1d(z1, self.weights)
        h2 = F.conv1d(z2, self.weights)
        h1 = torch.squeeze(h1, dim=1)
        h2 = torch.squeeze(h2, dim=1)
        dist = (h1 - h2 + margin).squeeze() / (zn12 + 1e-7)

        return 1.0 + dist

# VAELoss
class VAELoss(torch.nn.Module):

    def __init__(self, args):
        super(VAELoss, self).__init__()
        self.args = args
        self.lambda1 = torch.tensor(args.lambda1).cuda()
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, vae, x_stage, x_mmse, x_age, z_mu, z_logvar, x_hat_stage, x_hat_mmse, x_hat_age):

        # Reconstruction Loss
        recon_loss = 0
        recon_loss += F.cosine_embedding_loss(x_hat_stage, x_stage, torch.Tensor([1]).cuda(), reduction="sum")  # Clinical Labels
        recon_loss += F.cross_entropy(x_hat_stage, x_stage.argmax(1), reduction="sum")                          # Clinical Labels
        recon_loss += self.mae(x_hat_mmse, x_mmse)  # MMSE
        recon_loss += self.mse(x_hat_mmse, x_mmse)  # MMSE
        recon_loss += self.mae(x_hat_age, x_age)    # Age
        recon_loss += self.mse(x_hat_age, x_age)    # Age

        # KLDiv
        KLD_loss = - 0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), 1)

        # Regularization
        l1_regularization = torch.tensor(0).float().cuda()
        for name, param in vae.named_parameters():
            if 'bias' not in name:
                l1_regularization += torch.norm(param.cuda(), 1)

        # Take the average
        loss = recon_loss + torch.mean(KLD_loss) + self.args.lambda1 * l1_regularization

        return loss, recon_loss, torch.mean(KLD_loss)

# SOMLoss
class SOMLoss(torch.nn.Module):

    def __init__(self, args):
        super(SOMLoss, self).__init__()
        self.args = args

    def forward(self, weights, distances):
        # Calculate loss
        loss = torch.sum(weights * distances, 1).mean()
        return loss

class L1L2Loss(torch.nn.Module):

    def __init__(self, args):
        super(L1L2Loss, self).__init__()
        self.args = args
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')

    def forward(self, x_hat, x):
        # Reconstruction Loss
        recon_l1_loss = self.args.lambda_l1 * self.l1(x_hat, x)
        recon_l2_loss = self.args.lambda_l2 * self.l2(x_hat, x)

        loss = recon_l1_loss + recon_l2_loss

        return loss
