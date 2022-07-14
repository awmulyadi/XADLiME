import torch.optim as optim
import os
import datetime
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from models import ADPENVAE
from models import ADPENSOM
from models import ProgAE
from helpers import *

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--model", type=str, default='-')
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lambda1", type=float, default=0.0)
parser.add_argument("--lambda2", type=float, default=0.0001)
parser.add_argument("--lambda_l1", type=float, default=1.0)
parser.add_argument("--lambda_l2", type=float, default=1.0)
parser.add_argument("--weight_recon", type=float, default=10.0)
parser.add_argument("--dataset", type=str, default='adni_smri')
parser.add_argument("--load_images", type=bool, default=False)
parser.add_argument("--note", type=str, default='ProgAE')
args = parser.parse_args()

# GPU Configuration
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

args.vae_hiddens = [10, 16, 8, 3]
args.som_map_size = [1, 5, 20]
args.som_Tmin = 1.0 # Fixed
args.som_Tmax = 1.0 # Fixed
args.som_dim = 3
args.task = ['CN', 'SMCI', 'PMCI', 'AD']
args.spectrum_dim = 4

if (args.som_map_size[0]==1 and args.som_map_size[1]==5 and args.som_map_size[2]==20):
    directories_vaesom_finetune = ['-',
                                   '-',
                                   '-',
                                   '-',
                                   '-',]
log_dir_vaesom_finetune = directories_vaesom_finetune[args.fold-1]

# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = 'log/ProgAE/ProgAE_%s_GPU%s_%d/' % (date_str, args.gpu_id, args.fold)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(directory + 'map/')
    os.makedirs(directory + 'map/train/')
    os.makedirs(directory + 'map/valid/')
    os.makedirs(directory + 'map/test/')
    os.makedirs(directory + 'tflog/')
    os.makedirs(directory + 'model/')

# Text Logging
f = open(directory + 'setting.log', 'a')
writelog(f, '======================')
writelog(f, 'Load Images: %s' % args.load_images)
writelog(f, 'Model: %s' % args.model)
writelog(f, 'Lambda1: %.5f' % args.lambda1)
writelog(f, 'Lambda2: %.5f' % args.lambda2)
writelog(f, 'Lambda1: %.5f' % args.lambda_l1)
writelog(f, 'Lambda2: %.5f' % args.lambda_l2)
writelog(f, '----------------------')
writelog(f, 'SOM')
writelog(f, 'SOM MAP Size: %d x %d x %d' % (args.som_map_size[0], args.som_map_size[1], args.som_map_size[2]))
writelog(f, 'SOM Dim: %d' % args.som_dim)
writelog(f, 'SOM Tmin: %.5f' % args.som_Tmin)
writelog(f, 'SOM Tmax: %.5f' % args.som_Tmax)
writelog(f, 'Pretrained VAE: %s' % log_dir_vaesom_finetune)
writelog(f, 'Pretrained SOM: %s' % log_dir_vaesom_finetune)
writelog(f, 'SOM Dim Pretrained: %d' % args.som_dim)
writelog(f, '----------------------')
writelog(f, 'WEIGHT')
writelog(f, 'Weight Recon: %.5f' % args.weight_recon)
writelog(f, '----------------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Batch Size: %d' % args.batch_size)
writelog(f, 'Epoch: %d' % args.epoch)
writelog(f, '----------------------')
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
dataloaders, means, stds, mins, maxs = smri_batchloader(args, is_quartet=False)
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

x_test = None
def train(dataloader, dir='.'):
    # Set mode as training
    vae.eval()
    som.eval()
    ae.train()

    # Define training variables
    loss_recon_ = 0
    n_samples = 0

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):

        # Data
        y = xbatch['label'].long().to(device)
        d = xbatch['demographic'].float().to(device)
        batch_size = d.shape[0]
        if batch_size != args.batch_size:
            continue
        d_stage = d[:, :4]                  # Clinical Stage, Dimensionality: 4
        d_mmse = d[:, 4].unsqueeze(1)       # MMSE, Dimensionality: 1
        d_age = d[:, 5].unsqueeze(1)        # Age, Dimensionality: 1

        optimizer.zero_grad()

        # Make progression map
        _, h, _, _, _, _ = vae(d_stage, d_mmse, d_age)
        batch_h = h.view(batch_size, 1, -1)
        batch_prototypes = som_prototypes.expand(batch_size, -1, -1)
        rho = torch.pow(torch.cdist(batch_h, batch_prototypes), 2).squeeze(1)
        rho = nn.Softmax(dim=1)(-rho / torch.std(-rho, dim=1).unsqueeze(1))
        rho = (rho - rho.min(1)[0].view(-1, 1)) / (rho.max(1)[0].view(-1, 1) - rho.min(1)[0].view(-1, 1))
        rho = rho.view(-1, args.som_map_size[0], args.som_map_size[1], args.som_map_size[2])

        # AE
        z_rho, rho_hat = ae(rho)

        # Calculating loss
        loss_recon = args.weight_recon * criterion_recon(rho_hat, rho.detach())

        loss_recon.backward()
        optimizer.step()

        loss_recon_ += loss_recon.item()

        n_samples += batch_size

    # Take average
    loss_recon_ = loss_recon_ / n_samples
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)

    # Tensorboard Loggingss
    info = {'loss_recon_': loss_recon_}

    for tag, value in info.items():
        with tfw_train.as_default():
            with tf.device('/cpu:0'):
                tf.summary.scalar(tag,value, step=epoch)

def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    vae.eval()
    som.eval()
    ae.eval()

    # Define training variables
    loss_recon_ = 0
    n_samples = 0

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):

            # Data
            y = xbatch['label'].long().to(device)
            d = xbatch['demographic'].float().to(device)
            batch_size = d.shape[0]
            if batch_size != args.batch_size:
                continue
            d_stage = d[:, :4]                  # Clinical Stage, Dimensionality: 4
            d_mmse = d[:, 4].unsqueeze(1)       # MMSE, Dimensionality: 1
            d_age = d[:, 5].unsqueeze(1)        # Age, Dimensionality: 1

            # Make progression map
            _, h, _, _, _, _ = vae(d_stage, d_mmse, d_age)
            batch_h = h.view(batch_size, 1, -1)
            batch_prototypes = som_prototypes.expand(batch_size, -1, -1)
            rho = torch.pow(torch.cdist(batch_h, batch_prototypes), 2).squeeze(1)
            rho = nn.Softmax(dim=1)(-rho / torch.std(-rho, dim=1).unsqueeze(1))
            rho = (rho - rho.min(1)[0].view(-1, 1)) / (rho.max(1)[0].view(-1, 1) - rho.min(1)[0].view(-1, 1))
            rho = rho.view(-1, args.som_map_size[0], args.som_map_size[1], args.som_map_size[2])

            # AE
            z_rho, rho_hat = ae(rho)

            # Calculating loss
            loss_recon = args.weight_recon * criterion_recon(rho_hat, rho.detach())

            loss_recon_ += loss_recon.item()

            n_samples += batch_size

    # Take average
    loss_recon_ = loss_recon_ / n_samples
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)

    # Tensorboard Loggingss
    info = {'loss_recon_': loss_recon_}

    for tag, value in info.items():
        if phase == 'valid':
            with tfw_valid.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)
        else:
            with tfw_test.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)

    return loss_recon_

# Define Model
vae = ADPENVAE(args).to(device)
vae.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENVAE.pt' % (log_dir_vaesom_finetune)))
som = ADPENSOM(args).to(device)
som.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENSOM.pt' % (log_dir_vaesom_finetune)))
som_prototypes = change_view_3D(som.prototypes, args, transpose=True)
ae = ProgAE(args).to(device)
criterion_recon = nn.MSELoss().to(device)
optimizer = optim.Adam(list(ae.parameters()), lr=args.lr, weight_decay=args.lambda2)

# Define data type
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# Best epoch checking
valid = {'epoch': 0, 'loss': 99999}

# Train Epoch
for epoch in range(args.epoch):

    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, 'Training')
    train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    loss = evaluate('valid', dataloaders['valid'], dir=directory)

    # Save Model
    if loss < valid['loss']:
        torch.save(ae.state_dict(), directory + '/model/ProgAE.pt')
        writelog(f, 'Lowest validation loss is found! Validation Loss : %f' % loss)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss
        valid['epoch'] = epoch
        f2 = open(directory + 'setting.log', 'a')
        writelog(f2, 'Best Epoch %d' % epoch)
        f2.close()

writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

writelog(f, 'END OF TRAINING')
f.close()
