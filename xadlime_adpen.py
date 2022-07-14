import torch.optim as optim
import os
import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from models import ADPENVAE
from models import ADPENSOM
from losses import *
from helpers import *

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lambda1", type=float, default=0.00001)
parser.add_argument("--lambda2", type=float, default=0.0001)
parser.add_argument("--som_start_epoch", type=float, default=0)
parser.add_argument("--som_end_epoch", type=float, default=500)
# parser.add_argument("--som_map_size", nargs='+', type=int, default=[1, 1, 100])   # 1D Topology
parser.add_argument("--som_map_size", nargs='+', type=int, default=[1, 5, 20])      # 2D Topology
# parser.add_argument("--som_map_size", nargs='+', type=int, default=[2, 2, 25])    # 3D Topology
# parser.add_argument("--som_Tmin", type=float, default=2.0)
# parser.add_argument("--som_Tmax", type=float, default=20.0)
parser.add_argument("--som_dim", type=int, default=3)
parser.add_argument("--som_decay", type=str, default='exponential')
parser.add_argument("--z_dim", type=int, default=3)
parser.add_argument("--weight_vae", type=float, default=1.0)
parser.add_argument("--weight_order", type=float, default=1.0)
parser.add_argument("--weight_som", type=float, default=0.01)
parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default='adni_smri')
parser.add_argument("--load_images", type=bool, default=False)
parser.add_argument("--finetune", type=int, default=0) # 0: False (Pretrain), 1: True (Finetune)
parser.add_argument("--note", type=str, default='ADPEN')
args = parser.parse_args()

args.vae_hiddens = [10, 16, 8, 3]
args.task = ['CN', 'SMCI', 'PMCI', 'AD']
args.spectrum_dim = 4
args.batch_size = 4

if args.finetune:
    args.som_Tmin = 0.8
    args.som_Tmax = 2.0
    args.weight_som = 1.0
    if (args.som_map_size[0]==1 and args.som_map_size[1]==5 and args.som_map_size[2]==20):
        directories_vaesom = ['-',
                              '-',
                              '-',
                              '-',
                              '-',]
    log_dir_vaesom = directories_vaesom[args.fold-1]
else:
    args.som_Tmin = 2.0
    args.som_Tmax = 20.0

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
if args.finetune:
    directory = 'log/ADPEN/ADPEN_FINETUNE_%s_GPU%s_%d/' % (date_str, args.gpu_id, args.fold)
else:
    directory = 'log/ADPEN/ADPEN_%s_GPU%s_%d/' % (date_str, args.gpu_id, args.fold)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + 'img/')
    os.makedirs(directory + 'img/train/')
    os.makedirs(directory + 'img/valid/')
    os.makedirs(directory + 'img/test/')
    os.makedirs(directory + 'tflog/')
    os.makedirs(directory + 'model/')

# Text Logging
f = open(directory + 'setting.log', 'a')
writelog(f, '======================')
writelog(f, 'Lambda1: %.5f' % args.lambda1)
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, '----------------------')
writelog(f, 'SOM')
writelog(f, 'SOM MAP Size: %d x %d x %d' % (args.som_map_size[0], args.som_map_size[1], args.som_map_size[2]))
writelog(f, 'SOM Dim: %d' % args.som_dim)
writelog(f, 'SOM Distance Metric: Euclidean')
writelog(f, 'SOM Tmin: %.5f' % args.som_Tmin)
writelog(f, 'SOM Tmax: %.5f' % args.som_Tmax)
writelog(f, 'SOM Start Epoch: %d' % args.som_start_epoch)
writelog(f, 'SOM End Epoch: %d' % args.som_end_epoch)
writelog(f, 'SOM Decay: %s' % args.som_decay)
for h in args.vae_hiddens:
    writelog(f, 'VAE Hidden Units: %d' % h)
writelog(f, 'Z Dim: %.5f' % args.z_dim)
writelog(f, '----------------------')
writelog(f, 'WEIGHT')
writelog(f, 'Weight VAE: %.5f' % args.weight_vae)
writelog(f, 'Weight SOM: %.5f' % args.weight_som)
writelog(f, 'Weight Order: %.5f' % args.weight_order)
writelog(f, 'Margin: %.5f' % args.margin)
writelog(f, '----------------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '----------------------')
writelog(f, 'Finetune: %d' % args.finetune)
writelog(f, 'Note: %s' % args.note)
writelog(f, '======================')
f.close()

f = open(directory + 'log.log', 'a')
# Tensorboard Logging
with tf.device('/cpu:0'):
    tfw_train = tf.summary.create_file_writer(directory + 'tflog/kfold_' + str(args.fold) + '/train_')
    tfw_valid = tf.summary.create_file_writer(directory + 'tflog/kfold_' + str(args.fold) + '/valid_')
    tfw_test = tf.summary.create_file_writer(directory + 'tflog/kfold_' + str(args.fold) + '/test_')

# Tensor Seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define Loaders
writelog(f, 'Load data')
if args.finetune:
    dataloaders, means, stds, mins, maxs = smri_batchloader(args, is_quartet=False)
else:
    dataloaders, means, stds, mins, maxs = smri_batchloader(args, is_quartet=True)
f2 = open(directory + 'setting.log', 'a')
writelog(f2, 'MMSE Mean %.5f' % means[0])
writelog(f2, 'MMSE STD %.5f' % stds[0])
writelog(f2, 'MMSE Min %.5f' % mins[0])
writelog(f2, 'MMSE Max %.5f' % maxs[0])
writelog(f2, 'Age Mean %.5f' % means[1])
writelog(f2, 'Age STD %.5f' % stds[1])
writelog(f2, 'Age Min %.5f' % mins[1])
writelog(f2, 'Age Max %.5f' % maxs[1])
f2.close()

def train(dataloader, directory='.'):
    # Set mode as training
    if args.finetune:
        vae.eval()
        lssl.eval()
    else:
        vae.train()
        lssl.train()
    som.train()

    # Define training variables
    loss_total_ = 0
    loss_vae_ = 0
    loss_recon_ = 0
    loss_kld_ = 0
    loss_order_ = 0
    loss_som_ = 0
    n_samples = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):

        # Data
        x = xbatch['demographic'].float().to(device)
        batch_size = x.shape[0]
        if batch_size != args.batch_size:
            continue
        x_stage = x[:, :4]              # Clinical Stage, Dimensionality: 4
        x_mmse = x[:, 4].unsqueeze(1)   # MMSE, Dimensionality: 1
        x_age = x[:, 5].unsqueeze(1)    # Age, Dimensionality: 1

        # TRAIN VAESOM
        optimizer.zero_grad()
        z, z_mu, z_logvar, x_hat_stage, x_hat_mmse, x_hat_age  = vae(x_stage, x_mmse, x_age)
        loss_vae, loss_recon, loss_kld = criterion_vae(vae, x_stage, x_mmse, x_age,
                                                       z_mu, z_logvar,
                                                       x_hat_stage, x_hat_mmse, x_hat_age)
        loss_order = lssl(z[0, :].unsqueeze(0), z[1, :].unsqueeze(0), args.margin) + \
                     lssl(z[1, :].unsqueeze(0), z[2, :].unsqueeze(0), args.margin) + \
                     lssl(z[2, :].unsqueeze(0), z[3, :].unsqueeze(0), args.margin)
        w, distances = som(z, epoch - args.som_start_epoch, args.som_end_epoch - args.som_start_epoch)
        loss_som = criterion_som(w, distances)

        if args.finetune:
            loss_total = (args.weight_som * loss_som)
            loss_total.backward()
            optimizer.step()

            loss_total_ += loss_total.item()
            loss_vae_ += 0
            loss_recon_ += 0
            loss_kld_ += 0
            loss_order_ += 0
            loss_som_ += args.weight_som * loss_som.item()

        else:
            loss_total = (args.weight_vae * loss_vae) + \
                         (args.weight_order * loss_order) + \
                         (args.weight_som * loss_som)
            loss_total.backward()
            optimizer.step()

            loss_total_ += loss_total.item()
            loss_vae_ += args.weight_vae * loss_vae.item()
            loss_recon_ += loss_recon.item()
            loss_kld_ += loss_kld.item()
            loss_order_ += args.weight_order * loss_order.item()
            loss_som_ += args.weight_som * loss_som.item()

        n_samples += batch_size

    # Take average
    loss_total_ = loss_total_ / n_samples
    loss_vae_ = loss_vae_ / n_samples
    loss_recon_ = loss_recon_ / n_samples
    loss_kld_ = loss_kld_ / n_samples
    loss_order_ = loss_order_ / n_samples
    loss_som_ = loss_som_ / n_samples
    writelog(f, 'Loss Total: %.8f' % loss_total_)
    writelog(f, 'Loss VAE: %.8f' % loss_vae_)
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)
    writelog(f, 'Loss KLD: %.8f' % loss_kld_)
    writelog(f, 'Loss Order: %.8f' % loss_order_)
    writelog(f, 'Loss SOM: %.8f' % loss_som_)

    # Tensorboard Logging
    info = {'loss_total': loss_total_,
            'loss_vae': loss_vae_,
            'loss_recon': loss_recon_,
            'loss_kld': loss_kld_,
            'loss_order': loss_order_,
            'loss_som': loss_som_,
            'SOM.T': som.T}

    for tag, value in info.items():
        with tfw_train.as_default():
            with tf.device('/cpu:0'):
                tf.summary.scalar(tag,value, step=epoch)
                # tfw_train.flush()

def evaluate(phase, dataloader, directory='.'):
    # Set mode as training
    vae.eval()
    lssl.eval()
    som.eval()

    # Define training variables
    loss_total_ = 0
    loss_vae_ = 0
    loss_recon_ = 0
    loss_kld_ = 0
    loss_order_ = 0
    loss_som_ = 0
    n_samples = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):

            # Data
            x = xbatch['demographic'].float().to(device)
            batch_size = x.shape[0]
            if batch_size != args.batch_size:
                continue
            x_stage = x[:, :4]                  # Clinical Stage, Dimensionality: 4
            x_mmse = x[:, 4].unsqueeze(1)       # MMSE, Dimensionality: 1
            x_age = x[:, 5].unsqueeze(1)        # Age, Dimensionality: 1

            # TRAIN VAESOM
            z, z_mu, z_logvar, x_hat_stage, x_hat_mmse, x_hat_age  = vae(x_stage, x_mmse, x_age)
            loss_vae, loss_recon, loss_kld = criterion_vae(vae, x_stage, x_mmse, x_age,
                                                           z_mu, z_logvar,
                                                           x_hat_stage, x_hat_mmse, x_hat_age)
            loss_order = lssl(z_mu[0, :].unsqueeze(0), z_mu[1, :].unsqueeze(0), args.margin) + \
                         lssl(z_mu[1, :].unsqueeze(0), z_mu[2, :].unsqueeze(0), args.margin) + \
                         lssl(z_mu[2, :].unsqueeze(0), z_mu[3, :].unsqueeze(0), args.margin)
            w, distances = som(z_mu, epoch - args.som_start_epoch, args.som_end_epoch - args.som_start_epoch)
            loss_som = criterion_som(w, distances)

            if args.finetune:
                loss_total = (args.weight_som * loss_som)

                loss_total_ += loss_total.item()
                loss_vae_ += 0
                loss_recon_ += 0
                loss_kld_ += 0
                loss_order_ += 0
                loss_som_ += args.weight_som * loss_som.item()

            else:
                loss_total = (args.weight_vae * loss_vae) + \
                             (args.weight_order * loss_order) + \
                             (args.weight_som * loss_som)

                loss_total_ += loss_total.item()
                loss_vae_ += args.weight_vae * loss_vae.item()
                loss_recon_ += loss_recon.item()
                loss_kld_ += loss_kld.item()
                loss_order_ += args.weight_order * loss_order.item()
                loss_som_ += args.weight_som * loss_som.item()

            n_samples += batch_size

    # Take average
    loss_total_ = loss_total_ / n_samples
    loss_vae_ = loss_vae_ / n_samples
    loss_recon_ = loss_recon_ / n_samples
    loss_kld_ = loss_kld_ / n_samples
    loss_order_ = loss_order_ / n_samples
    loss_som_ = loss_som_ / n_samples
    writelog(f, 'Loss Total: %.8f' % loss_total_)
    writelog(f, 'Loss VAE: %.8f' % loss_vae_)
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)
    writelog(f, 'Loss KLD: %.8f' % loss_kld_)
    writelog(f, 'Loss Order: %.8f' % loss_order_)
    writelog(f, 'Loss SOM: %.8f' % loss_som_)

    # Tensorboard Logging
    info = {'loss_total': loss_total_,
            'loss_vae': loss_vae_,
            'loss_recon': loss_recon_,
            'loss_kld': loss_kld_,
            'loss_order': loss_order_,
            'loss_som': loss_som_}

    for tag, value in info.items():
        if phase == 'valid':
            with tfw_valid.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)
                    # tfw_valid.flush()
        else:
            with tfw_test.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)
                    # tfw_test.flush()

    return loss_total_

# Define Model
vae = ADPENVAE(args).to(device)
som = ADPENSOM(args).to(device)
if args.finetune:
    vae.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENVAE.pt' % (log_dir_vaesom)))
    som.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENSOM.pt' % (log_dir_vaesom)))
lssl = LSSLLoss(args.z_dim).to(device)
criterion_vae = VAELoss(args).to(device)
criterion_som = SOMLoss(args).to(device)

if args.finetune:
    params = list(som.parameters())
else:
    params = list(vae.parameters()) + list(lssl.parameters()) + list(som.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.lambda2)

# Best epoch checking
valid = {'epoch': 0, 'loss': 99999}

# Train Epoch
for epoch in range(args.epoch):
    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, 'Training')
    train(dataloaders['train'], directory=directory)

    writelog(f, 'Validation')
    loss = evaluate('valid', dataloaders['valid'], directory=directory)

    # Save Model
    if loss < valid['loss']:
        torch.save(vae.state_dict(), directory + '/model/ADPENVAE.pt')
        torch.save(lssl.state_dict(), directory + '/model/LSSLLoss.pt')
        torch.save(som.state_dict(), directory + '/model/ADPENSOM.pt')
        writelog(f, 'Lowest validation loss is found! Validation Loss : %f' % loss)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss
        valid['epoch'] = epoch
        f2 = open(directory + 'setting.log', 'a')
        writelog(f2, 'Best Epoch %d' % epoch)
        f2.close()

    writelog(f, 'SOM T %.3f' % som.T)

writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

writelog(f, 'Testing')
loss = evaluate('test', dataloaders['test'], directory=directory)

writelog(f, 'END OF TRAINING')
f.close()
