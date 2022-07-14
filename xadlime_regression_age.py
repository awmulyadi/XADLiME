import torch.optim as optim
import os
import datetime
import pickle
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from models import ADPENVAE
from models import ADPENSOM
from models import ProgAE
from models import XADLiMESynth
from models import XADLiMERegressor
from losses import *
from helpers import *

# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--lambda1", type=float, default=0.00001)
parser.add_argument("--lambda2", type=float, default=0.0001)
parser.add_argument("--lambda_l1", type=float, default=10.0)
parser.add_argument("--lambda_l2", type=float, default=10.0)
parser.add_argument("--weight_cons", type=float, default=1.0)
parser.add_argument("--weight_reg", type=float, default=1000.0)
parser.add_argument("--hidden_dim", type=int, default=1024)
parser.add_argument("--regressor_dim", type=int, default=64)
parser.add_argument("--lr_decay", type=float, default=0.98)
parser.add_argument("--lr_decay_epoch_start", type=int, default=0)
parser.add_argument("--lr_decay_step", type=int, default=10)
parser.add_argument("--class_scenario", type=str, default='-')
parser.add_argument("--task", nargs='+', type=str, default=['CN'])
parser.add_argument("--dataset", type=str, default='adni_smri')
parser.add_argument("--load_images", type=bool, default=True)
parser.add_argument("--note", type=str, default='XADLiME, Age Prediction')
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
args.spectrum_dim = 4

if (args.som_map_size[0]==1 and args.som_map_size[1]==5 and args.som_map_size[2]==20):
    args.regressor_input_dim = 64 * 16
    args.scenario = '2D'
    # Require 3 trained model:
    # - ADPEN Finetuned (VAE, SOM)
    # - ADPEN ProgAE
    directories_vaesom_finetune = ['-',
                                   '-',
                                   '-',
                                   '-',
                                   '-',]
    directories_progae = ['-',
                          '-',
                          '-',
                          '-',
                          '-',]
log_dir_vaesom_finetune = directories_vaesom_finetune[args.fold-1]
log_dir_progae = directories_progae[args.fold-1]

args.z_dim = 512*3*4*3
args.z_dim_map = [1, 512, 3, 4, 3]
args.rho_dim = args.som_map_size[0] * args.som_map_size[1]

# Logging purpose
date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
directory = 'log/XADLiME/%s/%s/XADLiME_%s_GPU%s_%d/' % (args.scenario, 'REG_AGE', date_str, args.gpu_id, args.fold)
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
writelog(f, 'Class Scenario: %s' % ('/').join(args.task))
writelog(f, 'Dataset: %s' % args.dataset)
writelog(f, 'Load Images: %s' % args.load_images)
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
writelog(f, '----------------------')
writelog(f, 'Pretrained VAE: %s' % log_dir_vaesom_finetune)
writelog(f, 'Pretrained SOM: %s' % log_dir_vaesom_finetune)
writelog(f, 'Pretrained Demo Progression AE: %s' % log_dir_progae)
writelog(f, 'SOM Dim Pretrained: %d' % args.som_dim)
writelog(f, 'Regressor Input Dim: %d' % args.regressor_input_dim)
writelog(f, 'Regressor Dim: %d' % args.regressor_dim)
writelog(f, '----------------------')
writelog(f, 'WEIGHT')
writelog(f, 'weight_cons: %.5f' % args.weight_cons)
writelog(f, 'weight_reg: %.5f' % args.weight_reg)
writelog(f, '----------------------')
writelog(f, 'Fold: %d' % args.fold)
writelog(f, 'Learning Rate: %.5f' % args.lr)
writelog(f, 'Learning Rate Decay: %.5f' % args.lr_decay)
writelog(f, 'Learning Rate Decay Step: %d' % args.lr_decay_step)
writelog(f, 'Learning Rate Decay Epoch Start: %d' % args.lr_decay_epoch_start)
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
dataloaders, means, stds, mins, maxs = smri_batchloader(args, is_quartet=False, seed=0)
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

def train(dataloader, dir='.'):
    # Set mode as training
    vae.eval()
    som.eval()
    ae.eval()
    synth.train()
    regressor.train()

    # Define variables
    loss_overall_ = 0
    loss_recon_ = 0
    loss_l1_ = 0
    loss_l2_ = 0
    loss_cons_ = 0
    loss_reg_ = 0
    n_samples = 0
    y_gts = np.array([]).reshape(0)
    y_hats = np.array([]).reshape(0)

    # Loop over the minibatch
    for i, xbatch in enumerate(tqdm(dataloader)):

        # Data
        x = normalize_gauss_aug_on(device, xbatch['data'])
        idx = xbatch['image_id']
        d = xbatch['demographic'].float().to(device)
        y = ((d[:, 5] * stds[1]) + means[1]) / 100.0 # Age
        y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten() * 100.0])
        batch_size = x.shape[0]
        d_stage = d[:, :4]                  # Clinical Stage, Dimensionality: 4
        d_mmse = d[:, 4].unsqueeze(1)       # MMSE, Dimensionality: 1
        d_age = d[:, 5].unsqueeze(1)        # Age, Dimensionality: 1

        optimizer.zero_grad()

        # Make progression map
        _, h, _, _, _, _ = vae(d_stage, d_mmse, d_age)
        h = h.view(batch_size, 1, -1)
        batch_prototypes = som_prototypes.expand(batch_size, -1, -1)
        rho = torch.pow(torch.cdist(h, batch_prototypes), 2).squeeze(1)
        rho = nn.Softmax(dim=1)(-rho / torch.std(-rho, dim=1).unsqueeze(1))
        rho = (rho - rho.min(1)[0].view(-1, 1)) / (rho.max(1)[0].view(-1, 1) - rho.min(1)[0].view(-1, 1))
        rho = rho.view(-1, args.som_map_size[0], args.som_map_size[1], args.som_map_size[2])
        z_rho, _ = ae(rho.detach())

        # Generate the distance map given sMRI images
        rho_hat = synth(x)
        z_rho_hat, _ = ae(rho_hat)
        y_logit, y_hat = regressor(z_rho_hat)

        # Calculating loss
        loss_cons = args.weight_cons * criterion_cons(z_rho_hat, z_rho.detach())
        loss_recon = criterion_recon(rho_hat, rho.detach())
        loss_reg = args.weight_reg * criterion_reg(y_hat.flatten(), y)

        # Regularization
        l1_regularization = torch.tensor(0).float().cuda()
        for name, param in synth.named_parameters():
            if 'bias' not in name:
                l1_regularization += param.abs().sum()
        loss_l1 = args.lambda1 * l1_regularization

        # Overall Loss
        loss_overall = loss_recon + loss_cons + loss_reg + loss_l1
        loss_overall.backward()
        optimizer.step()

        loss_overall_ += loss_overall.item()
        loss_recon_ += loss_recon.item()
        loss_cons_ += loss_cons.item()
        loss_reg_ += loss_reg.item()
        loss_l1_ += loss_l1.item()

        n_samples += batch_size

        # Collect the prediction label
        y_hat = y_hat.flatten().to('cpu').detach().numpy() * 100.0
        if i == 0:
            y_hats = y_hat
        else:
            y_hats = np.hstack([y_hats, y_hat])

    # Take average
    loss_overall_ = loss_overall_ / n_samples
    loss_recon_ = loss_recon_ / n_samples
    loss_l1_ = loss_l1_ / n_samples
    loss_l2_ = loss_l2_ / n_samples
    loss_cons_ = loss_cons_ / n_samples
    loss_reg_ = loss_reg_ / n_samples
    writelog(f, 'Loss Overall: %.8f' % loss_overall_)
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)
    writelog(f, 'Loss L1: %.8f' % loss_l1_)
    writelog(f, 'Loss L2: %.8f' % loss_l2_)
    writelog(f, 'Loss Consistency: %.8f' % loss_cons_)
    writelog(f, 'Loss CE: %.8f' % loss_reg_)

    # Tensorboard Loggingss
    info = {'loss_overall_': loss_overall_,
            'loss_recon': loss_recon_,
            'loss_l1_': loss_l1_,
            'loss_l2_': loss_l2_,
            'loss_cons_': loss_cons_,
            'loss_reg_': loss_reg_}

    # Regression Performance
    metric = calculate_performance_reg(y_gts, y_hats, args)
    metric_str = ['mse',
                  'rmse',
                  'mae',
                  'r2']
    for m in metric:
        writelog(f, m)
        for ms, mv in zip(metric_str, metric[m]):
            writelog(f, '%s: %.5f' % (ms, mv))
            info['%s_%s' % (m, ms)] = mv
        writelog(f, '--')

    for tag, value in info.items():
        with tfw_train.as_default():
            with tf.device('/cpu:0'):
                tf.summary.scalar(tag,value, step=epoch)

def evaluate(phase, dataloader, dir='.'):
    # Set mode as training
    vae.eval()
    som.eval()
    ae.eval()
    synth.eval()
    regressor.eval()

    # Define variables
    loss_overall_ = 0
    loss_recon_ = 0
    loss_l1_ = 0
    loss_l2_ = 0
    loss_cons_ = 0
    loss_reg_ = 0
    n_samples = 0
    y_gts = np.array([]).reshape(0)
    y_hats = np.array([]).reshape(0)

    # No Grad
    with torch.no_grad():
        # Loop over the minibatch
        for i, xbatch in enumerate(tqdm(dataloader)):

            # Data
            x = normalize_gauss_aug_on(device, xbatch['data'])
            idx = xbatch['image_id']
            d = xbatch['demographic'].float().to(device)
            y = ((d[:, 5] * stds[1]) + means[1]) / 100.0 # Age
            y_gts = np.hstack([y_gts, y.to('cpu').detach().numpy().flatten() * 100.0])
            batch_size = x.shape[0]
            d_stage = d[:, :4]                  # Clinical Stage, Dimensionality: 4
            d_mmse = d[:, 4].unsqueeze(1)       # MMSE, Dimensionality: 1
            d_age = d[:, 5].unsqueeze(1)        # Age, Dimensionality: 1

            # Make progression map
            _, h, _, _, _, _ = vae(d_stage, d_mmse, d_age)
            h = h.view(batch_size, 1, -1)
            batch_prototypes = som_prototypes.expand(batch_size, -1, -1)
            rho = torch.pow(torch.cdist(h, batch_prototypes), 2).squeeze(1)
            rho = nn.Softmax(dim=1)(-rho / torch.std(-rho, dim=1).unsqueeze(1))
            rho = (rho - rho.min(1)[0].view(-1, 1)) / (rho.max(1)[0].view(-1, 1) - rho.min(1)[0].view(-1, 1))
            rho = rho.view(-1, args.som_map_size[0], args.som_map_size[1], args.som_map_size[2])
            z_rho, _ = ae(rho.detach())

            # Generate the distance map given sMRI images
            rho_hat = synth(x)
            z_rho_hat, _ = ae(rho_hat)
            y_logit, y_hat = regressor(z_rho_hat)

            # Calculating loss
            loss_cons = args.weight_cons * criterion_cons(z_rho_hat, z_rho.detach())
            loss_recon = criterion_recon(rho_hat, rho.detach())
            loss_reg = args.weight_reg * criterion_reg(y_hat.flatten(), y)

            # Regularization
            l1_regularization = torch.tensor(0).float().cuda()
            for name, param in synth.named_parameters():
                if 'bias' not in name:
                    l1_regularization += param.abs().sum()
            loss_l1 = args.lambda1 * l1_regularization

            # Overall Loss
            loss_overall = loss_recon + loss_cons + loss_reg + loss_l1

            loss_overall_ += loss_overall.item()
            loss_recon_ += loss_recon.item()
            loss_cons_ += loss_cons.item()
            loss_reg_ += loss_reg.item()
            loss_l1_ += loss_l1.item()

            n_samples += batch_size

            # Collect the prediction label
            y_hat = y_hat.flatten().to('cpu').detach().numpy() * 100.0
            if i == 0:
                y_hats = y_hat
            else:
                y_hats = np.hstack([y_hats, y_hat])

    # Take average
    loss_overall_ = loss_overall_ / n_samples
    loss_recon_ = loss_recon_ / n_samples
    loss_l1_ = loss_l1_ / n_samples
    loss_l2_ = loss_l2_ / n_samples
    loss_cons_ = loss_cons_ / n_samples
    loss_reg_ = loss_reg_ / n_samples
    writelog(f, 'Loss Overall: %.8f' % loss_overall_)
    writelog(f, 'Loss Recon: %.8f' % loss_recon_)
    writelog(f, 'Loss L1: %.8f' % loss_l1_)
    writelog(f, 'Loss L2: %.8f' % loss_l2_)
    writelog(f, 'Loss Consistency: %.8f' % loss_cons_)
    writelog(f, 'Loss CE: %.8f' % loss_reg_)

    # Tensorboard Loggingss
    info = {'loss_overall_': loss_overall_,
            'loss_recon': loss_recon_,
            'loss_l1_': loss_l1_,
            'loss_l2_': loss_l2_,
            'loss_cons_': loss_cons_,
            'loss_reg_': loss_reg_}

    # Regression Performance
    metric = calculate_performance_reg(y_gts, y_hats, args)
    metric_str = ['mse',
                  'rmse',
                  'mae',
                  'r2']
    for m in metric:
        writelog(f, m)
        for ms, mv in zip(metric_str, metric[m]):
            writelog(f, '%s: %.5f' % (ms, mv))
            info['%s_%s' % (m, ms)] = mv
        writelog(f, '--')


    for tag, value in info.items():
        if phase == 'valid':
            with tfw_valid.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)
        else:
            with tfw_test.as_default():
                with tf.device('/cpu:0'):
                    tf.summary.scalar(tag, value, step=epoch)

    return loss_overall_, metric[('/').join(args.task)][0]

# Define Model
vae = ADPENVAE(args).to(device)
vae.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENVAE.pt' % (log_dir_vaesom_finetune)))
som = ADPENSOM(args).to(device)
som.load_state_dict(torch.load('log/ADPEN/%s/model/ADPENSOM.pt' % (log_dir_vaesom_finetune)))
# decoded = pickle.load(open('log/ADPEN/%s/decoded_prototypes.pickle' % (log_dir_vaesom_finetune), 'rb'))
som_prototypes = change_view_3D(som.prototypes, args, transpose=True)

ae = ProgAE(args).to(device)
ae.load_state_dict(torch.load('log/ProgAE/%s/model/ProgAE.pt' % (log_dir_progae)))

synth = XADLiMESynth(args).to(device)
regressor = XADLiMERegressor(args, n_output=1).to(device)
criterion_recon = L1L2Loss(args).to(device)
criterion_cons = L1L2Loss(args).to(device)
criterion_reg = nn.MSELoss().to(device)
params = list(synth.parameters()) + list(regressor.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.lambda2)

# Best epoch checking
valid = {'epoch': 0, 'loss': 99999, 'mse': 99999}

# Train Epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
for epoch in range(args.epoch):

    writelog(f, '--- Epoch %d' % epoch)
    writelog(f, 'Training')
    train(dataloaders['train'], dir=directory)

    writelog(f, 'Validation')
    loss, mse = evaluate('valid', dataloaders['valid'], dir=directory)

    # Save Model
    if mse <= valid['mse']:
        torch.save(synth.state_dict(), directory + '/model/XADLiMESynth.pt')
        torch.save(regressor.state_dict(), directory + '/model/XADLiMERegressor.pt')
        writelog(f, 'Best validation MSE is found! Validation MSE : %f' % mse)
        writelog(f, 'Models at Epoch %d are saved!' % epoch)
        valid['loss'] = loss
        valid['epoch'] = epoch
        valid['mse'] = mse
        f2 = open(directory + 'setting.log', 'a')
        writelog(f2, 'Best Epoch %d' % epoch)
        f2.close()

    writelog(f, 'Test')
    _, _ = evaluate('test', dataloaders['test'], dir=directory)

    # Learning Rate
    if epoch >= args.lr_decay_epoch_start:
        writelog(f, 'Before Step Learning Rate %.8f' % get_lr(optimizer))
        scheduler.step()
        writelog(f, 'After Step Learning Rate %.8f' % get_lr(optimizer))

writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

# Define models and load trained parameters
synth = XADLiMESynth(args).to(device)
synth.load_state_dict(torch.load(directory + '/model/XADLiMESynth.pt'))
regressor = XADLiMERegressor(args, n_output=1).to(device)
regressor.load_state_dict(torch.load(directory + '/model/XADLiMERegressor.pt'))

writelog(f, 'Test')
_, _ = evaluate('test', dataloaders['test'], dir=directory)
writelog(f, directory)
writelog(f, 'END OF TRAINING')
f.close()
